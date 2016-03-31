#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This is a simple tool that converts CSV file into a JSON file.
# The first line of the input CSV file is used as the field names.
#
# Use 'OrderedDict' to keep the input fields order.
#
# Aaron LI
# 2015/06/11
#

from __future__ import print_function, division

__version__ = "0.1.0"
__date__ = "2015/06/11"

import sys
import argparse
import csv
import json

from collections import OrderedDict


def csv2json(csvfile, jsonfile=None):
    """
    Convert CSV data to JSON data.
    The first line of CSV data is used as the field names.

    Return:
        If jsonfile is None, then return a list of JSON dict.
    """
    if not hasattr(csvfile, "read"):
        csvfile = open(csvfile, "r")
    if (jsonfile is not None) and (not hasattr(jsonfile, "write")):
        jsonfile = open(jsonfile, "w")
    csvdata = list(csv.reader(csvfile))
    fieldnames = csvdata[0]
    # use 'OrderedDict' to keep fields order
    jsondata = [ OrderedDict(zip(fieldnames, row)) for row in csvdata[1:] ]
    csvfile.close()
    if jsonfile is None:
        return jsondata
    else:
        # 'ensure_ascii=False' to support UTF-8
        json.dump(jsondata, jsonfile, ensure_ascii=False, indent=4)
        jsonfile.close()


def main():
    # command line options & arguments
    parser = argparse.ArgumentParser(
            description="Simple CSV to JSON convertor")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("csvfile", help="Input CSV file")
    parser.add_argument("jsonfile", nargs="?", default=sys.stdout,
            help="Output JSON file (default stdout)")
    args = parser.parse_args()

    csv2json(args.csvfile, args.jsonfile)


if __name__ == "__main__":
    main()

