#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Calculate the mean values of the cube.
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the mean value of the data cube")
    parser.add_argument("-d", "--dtype", default="float32",
                        help="NumPy dtype of data cubes (default: float32)")
    parser.add_argument("infiles", nargs="+", help="input data cubes")
    args = parser.parse_args()

    print("# filename: mean side_length")
    for f in args.infiles:
        cube = np.fromfile(open(f, "rb"), dtype=args.dtype)
        sidelen = round(cube.shape[0] ** (1.0/3))
        mean = cube.mean()
        print("%s:\t%g\t\t%d" % (f, mean, sidelen))


if __name__ == "__main__":
    main()
