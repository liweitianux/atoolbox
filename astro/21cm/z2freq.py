#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Convert redshifts to the observed frequency of the 21 cm signal,
and vice versa.
"""

import os
import sys
import argparse

import numpy as np


prog = os.path.basename(sys.argv[0])
freq21cm = 1420405751.7667 / 1e6  # [MHz]


def z2freq(redshifts, print_=False):
    redshifts = np.asarray(redshifts)
    freqs = freq21cm / (redshifts + 1.0)
    if print_:
        print("# redshift  frequency[MHz]")
        for z, f in zip(redshifts, freqs):
            print("%.4f  %.2f" % (z, f))
    return freqs


def freq2z(freqs, print_=False):
    freqs = np.asarray(freqs)
    redshifts = freq21cm / freqs - 1.0
    if print_:
        print("# frequency[MHz]  redshift")
        for f, z in zip(freqs, redshifts):
            print("%.2f  %.4f" % (f, z))
    return redshifts


def parse_inputs(inputs):
    values = []
    for inp in inputs:
        try:
            v = float(inp)
            values += [v]
        except ValueError:
            begin, step, stop = inp.split(":")
            begin, step, stop = float(begin), float(step), float(stop)
            v = np.arange(start=begin, stop=stop+step/2, step=step)
            values += list(v)
    return values


def main():
    if prog == "z2freq.py":
        description = "Convert redshifts to observed 21 cm frequencies"
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("inputs", nargs="+",
                            help="input redshifts: <z1> <begin:step:stop>")
        args = parser.parse_args()
        redshifts = parse_inputs(args.inputs)
        z2freq(redshifts, print_=True)
    elif prog == "freq2z.py":
        description = "Convert observed 21 cm frequencies to redshifts"
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("inputs", nargs="+",
                            help="input frequencies [MHz]: <freq1> " +
                            "<begin:step:stop>")
        args = parser.parse_args()
        freqs = parse_inputs(args.inputs)
        freq2z(freqs, print_=True)
    else:
        raise RuntimeError("unknown program name: %s" % prog)


if __name__ == "__main__":
    main()
