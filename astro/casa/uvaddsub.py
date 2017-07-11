#!casa-script
# -*- mode: python -*-
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT License
#

"""
Copy the first MeasurementSet as the output MeasurementSet, and
subtract/add the DATA from the second MeasurementSet.

NOTE: The output MeasurementSet DATA column is altered directly.
"""

import sys
import argparse
import shutil

# from casac.ms import ms


def get_data(msname):
    print("Opening MS: %s ..." % msname)
    ms.open(msname, nomodify=True)
    ms.selectinit(datadescid=0)
    print("Geting MS data column ...")
    rec = ms.getdata(["data"])
    data = rec["data"]
    print("data mean:", data.mean())
    ms.close()
    print("Closed MS: %s" % msname)
    return data


def calc_ms(msname, data2, operation="sub"):
    print("Opening MS: %s ..." % msname)
    ms.open(msname, nomodify=False)
    ms.selectinit(datadescid=0)
    print("Geting MS data column ...")
    rec = ms.getdata(["data"])
    data1 = rec["data"]
    print("data1 mean:", data1.mean())
    print("data2 mean:", data2.mean())
    if operation == "sub":
        data_out = data1 - data2
    elif operation == "add":
        data_out = data1 + data2
    else:
        raise ValueError("invalid operation: %s" % operation)
    print("data_out mean:", data_out.mean())
    rec = {"data": data_out}
    print("Putting new data ...")
    ms.putdata(rec)
    ms.close()
    print("Closed MS: %s" % msname)


def main(argv):
    parser = argparse.ArgumentParser(description="MS addition/subtraction")
    parser.add_argument("--operation", dest="operation", required=True,
                        choices=["add", "sub"],
                        help="operation (add/sub) to be performed")
    parser.add_argument("--ms1", dest="ms1", required=True,
                        help="the first MeasurementSet")
    parser.add_argument("--ms2", dest="ms2", required=True,
                        help="the second MeasurementSet")
    exgrp = parser.add_mutually_exclusive_group(required=True)
    exgrp.add_argument("--inplace", dest="inplace", action="store_true",
                       help="modify the first MS in place")
    exgrp.add_argument("--ms-out", dest="msout",
                       help="output MeasurementSet name (copied from " +
                       "the first MS)")
    args = parser.parse_args(argv)

    if args.inplace:
        print("Modify MS1 in place!")
        msout = args.ms1
    else:
        print("Copying MS1 to be output MS ...")
        msout = args.msout
        shutil.copytree(args.ms1, msout)

    data2 = get_data(args.ms2)
    calc_ms(msout, data2, operation=args.operation)


if __name__ == "__main__":
    argi = sys.argv.index("--") + 1
    argv = sys.argv[argi:]
    print("argv:", argv)
    main(argv)
