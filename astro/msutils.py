#!/usr/bin/env python3
#
# Copyright (c) 2016-2017 Sphesihle Makhathini
# Copyright (c) 2017 Aaron LI
# GNU General Public License v2.0 (GPLv2)
#

"""
MSUtils - A set of CASA MeasurementSet manipulation tools
Based on: https://github.com/SpheMakh/msutils
"""

import argparse
from collections import OrderedDict
from pprint import pprint

import numpy as np
from casacore import tables
from casacore.tables import table, maketabdesc, makearrcoldesc


def getinfo(msname):
    """
    Summarize the basic information of a MS.

    Parameters
    ----------
    msname : str
        Name of the MS
    """
    tab = tables.table(msname, ack=False)

    info = OrderedDict([
        ("Ncol", tab.ncols()),
        ("Nrow", tab.nrows()),
        ("Ncor", tab.getcell("DATA", 0).shape[-1]),
        ("Info", tab.info()),
        ("Keywords", tab.getkeywords().keys()),
        ("Columns", tab.colnames()),
        ("ColKeywords", OrderedDict([
            (cname, tab.getcolkeywords(cname)) for cname in tab.colnames()
        ])),
        ("Exposure", tab.getcell("EXPOSURE", 0)),
        ("FIELD", OrderedDict()),
        ("SPW", OrderedDict()),
        ("SCAN", OrderedDict()),
    ])
    tabs = {
        "FIELD": tables.table(msname+"/FIELD", ack=False),
        "SPW":   tables.table(msname+"/SPECTRAL_WINDOW", ack=False),
    }

    field_ids = tabs["FIELD"].getcol("SOURCE_ID")
    info["FIELD"]["STATE_ID"] = [None]*len(field_ids)
    info["FIELD"]["PERIOD"] = [None]*len(field_ids)
    for fid in field_ids:
        ftab = tab.query("FIELD_ID=={0:d}".format(fid))
        state_id = ftab.getcol("STATE_ID")[0]
        info["FIELD"]["STATE_ID"][fid] = int(state_id)
        scans = {}
        total_length = 0
        for scan in set(ftab.getcol("SCAN_NUMBER")):
            stab = ftab.query("SCAN_NUMBER=={0:d}".format(scan))
            length = (stab.getcol("TIME").max() - stab.getcol("TIME").min())
            stab.close()
            scans[str(scan)] = length
            total_length += length

        info["SCAN"][str(fid)] = scans
        info["FIELD"]["PERIOD"][fid] = total_length
        ftab.close()

    for key, _tab in tabs.items():
        if key == "SPW":
            colnames = ["CHAN_FREQ", "MEAS_FREQ_REF",
                        "REF_FREQUENCY", "TOTAL_BANDWIDTH",
                        "NAME", "NUM_CHAN", "IF_CONV_CHAIN",
                        "NET_SIDEBAND", "FREQ_GROUP_NAME"]
        else:
            colnames = _tab.colnames()
        for name in colnames:
            try:
                info[key][name] = _tab.getcol(name).tolist()
            except AttributeError:
                info[key][name] = _tab.getcol(name)
        _tab.close()

    # Get the minimum and maximum baselines
    uv = tab.getcol("UVW")[:, :2]
    baselines = np.sqrt(np.sum(uv**2, axis=1))
    info["Baseline"] = {"min": baselines.min(), "max": baselines.max()}

    tab.close()
    return info


def addcol(msname, colname=None, shape=None,
           data_desc_type="array",
           valuetype=None,
           init_with=None,
           coldesc=None,
           coldmi=None,
           clone="DATA",
           rowchunk=None):
    """
    Add a column to MS

    Parameters
    ----------
    msanme : str
        MS to which to add the column
    colname : str
        Name of the column to be added
    shape : shape
    valuetype : data type
    data_desc_type :
        * ``scalar`` - scalar elements
        * ``array`` - array elements
    init_with : value to initialize the column with
    """
    tab = table(msname, readonly=False)

    if colname in tab.colnames():
        print("Column already exists")
        return "exists"

    print("Attempting to add %s column to %s" % (colname, msname))

    valuetype = valuetype or "complex"

    if coldesc:
        data_desc = coldesc
        shape = coldesc["shape"]
    elif shape:
        data_desc = maketabdesc(makearrcoldesc(colname,
                                               init_with,
                                               shape=shape,
                                               valuetype=valuetype))
    elif valuetype == "scalar":
        data_desc = maketabdesc(makearrcoldesc(colname,
                                               init_with,
                                               valuetype=valuetype))
    elif clone:
        element = tab.getcell(clone, 0)
        try:
            shape = element.shape
            data_desc = maketabdesc(makearrcoldesc(colname,
                                                   element.flatten()[0],
                                                   shape=shape,
                                                   valuetype=valuetype))
        except AttributeError:
            shape = []
            data_desc = maketabdesc(makearrcoldesc(colname,
                                                   element,
                                                   valuetype=valuetype))

    colinfo = [data_desc, coldmi] if coldmi else [data_desc]
    tab.addcols(*colinfo)

    print("Column added successfully.")

    if init_with is None:
        tab.close()
        return "added"
    else:
        spwids = set(tab.getcol("DATA_DESC_ID"))
        for spw in spwids:
            print("Initializing column {0}. DDID is {1}".format(colname, spw))
            tab_spw = tab.query("DATA_DESC_ID=={0:d}".format(spw))
            nrows = tab_spw.nrows()

            rowchunk = rowchunk or nrows/10
            dshape = [0] + [a for a in shape]
            for row0 in range(0, nrows, rowchunk):
                nr = min(rowchunk, nrows-row0)
                dshape[0] = nr
                print("Wrtiting to column  %s (rows %d to %d)" %
                      (colname, row0, row0+nr-1))
                dtype = init_with.dtype
                tab_spw.putcol(colname,
                               np.ones(dshape, dtype=dtype) * init_with,
                               row0, nr)
            tab_spw.close()
    tab.close()


def sumcols(msname, col1=None, col2=None, outcol=None, cols=None,
            subtract=False):
    """
    Add col1 to col2, or sum columns in "cols" list.

    Parameters
    ----------
    subtract : bool
        Subtract ``col2`` from ``col1``
    """
    tab = table(msname, readonly=False)
    if outcol not in tab.colnames():
        print("outcol {0:s} does not exist, will add it first.".format(outcol))
        addcol(msname, outcol, clone=col1 or cols[0])

    spws = set(tab.getcol("DATA_DESC_ID"))
    for spw in spws:
        tab_spw = tab.query("DATA_DESC_ID=={0:d}".format(spw))
        nrows = tab_spw.nrows()
        rowchunk = nrows//10 if nrows > 10000 else nrows
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows-row0)
            print("Wrtiting to column  %s (rows %d to %d)" %
                  (outcol, row0, row0+nr-1))
            if subtract:
                data = (tab_spw.getcol(col1, row0, nr) -
                        tab_spw.getcol(col2, row0, nr))
            else:
                cols = cols or [col1, col2]
                data = 0
                for col in cols:
                    data += tab.getcol(col, row0, nr)

            tab_spw.putcol(outcol, data, row0, nr)
        tab_spw.close()

    tab.close()


def copycol(msname, fromcol, tocol):
    """
        Copy data from one column to another
    """

    tab = table(msname, readonly=False)
    if tocol not in tab.colnames():
        addcol(msname, tocol, clone=fromcol)

    spws = set(tab.getcol("DATA_DESC_ID"))
    for spw in spws:
        tab_spw = tab.query("DATA_DESC_ID=={0:d}".format(spw))
        nrows = tab_spw.nrows()
        rowchunk = nrows//10 if nrows > 5000 else nrows
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows-row0)
            data = tab_spw.getcol(fromcol, row0, nr)
            tab_spw.putcol(tocol, data, row0, nr)

        tab_spw.close()
    tab.close()


def calc_vis_noise(msname, sefd, spw_id=0):
    """
    Calculate the nominal per-visibility noise
    """
    tab = table(msname)
    spwtab = table(msname + "/SPECTRAL_WINDOW")

    freq0 = spwtab.getcol("CHAN_FREQ")[spw_id, 0]
    wavelength = 300e+6/freq0
    bw = spwtab.getcol("CHAN_WIDTH")[spw_id, 0]
    dt = tab.getcol("EXPOSURE", 0, 1)[0]
    dtf = (tab.getcol("TIME", tab.nrows()-1, 1)-tab.getcol("TIME", 0, 1))[0]

    tab.close()
    spwtab.close()

    print("%s: frequency %.2f MHz (lambda=%.2fm)" %
          (msname, freq0/1e6, wavelength))
    print("%s: bandwidth %.2g kHz, %.2fs integration, %.2fh synthesis" %
          (bw*1e-3, dt, dtf/3600))
    noise = sefd / np.sqrt(abs(2*bw*dt))
    print("SEFD of %.2f Jy gives per-visibility noise of %.2f mJy" %
          (sefd, noise*1000))

    return noise


def addnoise(msname, column="MODEL_DATA",
             noise=0, sefd=551,
             rowchunk=None,
             addToCol=None,
             spw_id=None):
    """
    Add Gaussian noise to MS, given a stdandard deviation (noise).
    This noise can be also be calculated given SEFD value
    """

    tab = table(msname, readonly=False)

    multi_chan_noise = False
    if hasattr(noise, "__iter__"):
        multi_chan_noise = True
    elif hasattr(sefd, "__iter__"):
        multi_chan_noise = True
    else:
        noise = noise or calc_vis_noise(msname, sefd=sefd,
                                        spw_id=spw_id or 0)

    spws = set(tab.getcol("DATA_DESC_ID"))
    for spw in spws:
        tab_spw = tab.query("DATA_DESC_ID=={0:d}".format(spw))
        nrows = tab_spw.nrows()
        nchan, ncor = tab_spw.getcell("DATA", 0).shape
        rowchunk = rowchunk or nrows/10
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows-row0)
            data = (np.random.randn(nr, nchan, ncor) +
                    1j*np.random.randn(nr, nchan, ncor))
            if multi_chan_noise:
                noise = noise[np.newaxis, :, np.newaxis]
            data *= noise

            if addToCol:
                data += tab_spw.getcol(addToCol, row0, nr)
                print("%s + noise --> %s (rows %d to %d)" %
                      (addToCol, column, row0, row0+nr-1))
            else:
                print("Adding noise to column %s (rows %d to %d)" %
                      (column, row0, row0+nr-1))

            tab_spw.putcol(column, data, row0, nr)
        tab_spw.close()

    tab.close()


def cmd_info(args):
    """
    Sub-command: "info", show MS basic information
    """
    msname = args.msname
    info = getinfo(msname)
    pprint(info)


def main():
    parser = argparse.ArgumentParser(
        description="CASA MeasurementSet (MS) manipulation utilities")

    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")

    # sub-command: "info"
    parser_info = subparsers.add_parser("info", help="show MS basic info")
    parser_info.add_argument("msname", help="MS name")
    parser_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
