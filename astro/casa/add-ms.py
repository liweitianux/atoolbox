#!casa-script
# -*- mode: python -*-
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT License
#

"""
Add the ``DATA`` column from multiple MeasurementSets.
"""

import sys
import argparse
import shutil

# from casac.ms import ms


def get_data(msname):
    print("Opening MS: %s ..." % msname)
    ms.open(msname, nomodify=True)
    ms.selectinit(datadescid=0)
    print("Geting MS DATA column ...")
    rec = ms.getdata(["data"])
    data = rec["data"]
    print("Data mean:", data.mean())
    ms.close()
    print("Closed MS: %s" % msname)
    return data


def add_ms(msname, data2):
    print("Opening MS: %s ..." % msname)
    ms.open(msname, nomodify=False)
    ms.selectinit(datadescid=0)
    print("Geting MS DATA column ...")
    rec = ms.getdata(["data"])
    data1 = rec["data"]
    print("Data mean:", data1.mean())
    print("Adding data from other MS ...")
    for d2 in data2:
        data1 += d2
    print("Output data mean:", data1.mean())
    rec = {"data": data1}
    print("Putting new data ...")
    ms.putdata(rec)
    ms.close()
    print("Closed MS: %s" % msname)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Add DATA of multiple MeasurementSets")
    parser.add_argument("-i", "--input", dest="input", nargs="+",
                        help="two or more MeasurementSets to be added")
    parser.add_argument("-o", "--output", dest="output", required=True,
                        help="output MeasurementSet name")
    args = parser.parse_args(argv)

    nms = len(args.input)
    if nms < 2:
        raise RuntimeError("Two or more input MeasurementSets required")

    print("Copying the first input MS to be the output MS ...")
    ms1 = args.input[0]
    msout = args.output
    shutil.copytree(ms1, msout)

    data2 = []
    for msname in args.input[1:]:
        data2.append(get_data(msname))
    add_ms(msout, data2)


if __name__ == "__main__":
    argi = sys.argv.index("--") + 1
    argv = sys.argv[argi:]
    print("argv:", argv)
    main(argv)
