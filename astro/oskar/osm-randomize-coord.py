#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Create new randomized coordinates by adding random offset to the existing
OSKAR sky model (i.e., osm), and replace original coordinates with the
specified new ones.
"""

import os
import argparse

import numpy as np


class OskarSkyModel:
    """
    OSKAR sky model
    """
    def __init__(self, infile):
        self.filename = infile
        self.header = self.get_header(infile)
        self.load_data(infile)

    @staticmethod
    def get_header(infile):
        """
        Read the OSM header lines
        """
        header = []
        with open(infile) as f:
            for line in f.readlines():
                if line[0] == "#":
                    header.append(line)
                else:
                    break
        print("Read OSM header:\n%s" % "".join(header))
        return header

    def load_data(self, infile):
        try:
            data = np.loadtxt(infile)
        except ValueError:
            data = np.loadtxt(infile, delimiter=",")
        self.ra = data[:, 0]    # [deg]
        self.dec = data[:, 1]   # [deg]
        self.flux = data[:, 2]  # [Jy]
        self.number = len(self.flux)
        print("Loaded OSM data from file: %s" % infile)

    def randomize_coord(self, sigma):
        """
        Randomize the coordinates by adding an offset sampling
        from a Gaussian of sigma.
        """
        self.offset_sigma = sigma/3600.0  # [arcsec] -> [deg]
        print("Random offset: %.1f [arcsec] == %.6f [deg]" %
              (sigma, self.offset_sigma))
        self.ra += np.random.normal(loc=0.0, scale=self.offset_sigma,
                                    size=self.number)
        self.dec += np.random.normal(loc=0.0, scale=self.offset_sigma,
                                     size=self.number)
        print("Generated randomized coordinates")

    def replace_coord(self, coordfile):
        """
        Replace the coordinates with the data from the given
        coordinate file.
        """
        try:
            coord_new = np.loadtxt(coordfile)
        except ValueError:
            coord_new = np.loadtxt(coordfile, delimiter=",")
        ra_new = coord_new[:, 0]
        dec_new = coord_new[:, 1]
        if self.number != len(ra_new):
            raise RuntimeError("invalid coordinate file: %s" % coordfile)
        self.ra = ra_new
        self.dec = dec_new
        print("Replaced coordinates")

    def save_data(self, data, outfile, clobber=False):
        if os.path.exists(outfile) and not clobber:
            raise OSError("file already exists: %s" % outfile)
        with open(outfile, "wb") as fb:
            for line in self.header:
                fb.write(line.encode("utf-8"))
            np.savetxt(fb, data)

    def save_coord(self, outfile, clobber=False):
        data = np.column_stack([self.ra, self.dec])
        self.save_data(data=data, outfile=outfile, clobber=clobber)
        print("Wrote coordinates to file: %s" % outfile)

    def save_osm(self, outfile, clobber=False):
        data = np.column_stack([self.ra, self.dec, self.flux])
        self.save_data(data=data, outfile=outfile, clobber=clobber)
        print("Wrote OSM to file: %s" % outfile)


def cmd_create(args):
    """
    sub-command: create - create randomized coordinates
    """
    osm = OskarSkyModel(args.infile)
    osm.randomize_coord(sigma=args.sigma)
    osm.save_coord(outfile=args.coordfile, clobber=args.clobber)


def cmd_replace(args):
    """
    sub-command: create - create randomized coordinates
    """
    osm = OskarSkyModel(args.infile)
    osm.replace_coord(coordfile=args.coordfile)
    if not args.nobackup:
        backfile = args.infile + ".bak"
        os.rename(args.infile, backfile)
        print("Backed up OSM as: %s" % backfile)
    osm.save_osm(outfile=args.infile, clobber=True)


def main():
    parser = argparse.ArgumentParser(
        description="Randomize OSKAR sky model source coordinates")
    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")
    # sub-command: "create"
    parser_create = subparsers.add_parser(
        "create", help="create randomized coordinates")
    parser_create.add_argument("-C", "--clobber", dest="clobber",
                               action="store_true",
                               help="overwrite existing output file")
    parser_create.add_argument("-s", "--sigma", dest="sigma",
                               required=True, type=float,
                               help="random offset sigma [arcsec]")
    parser_create.add_argument("-c", "--coord-file", dest="coordfile",
                               required=True,
                               help="output coordinate file")
    parser_create.add_argument("infile", help="input OSKAR sky model")
    parser_create.set_defaults(func=cmd_create)
    # sub-command: "replace"
    parser_replace = subparsers.add_parser(
        "replace", help="replace coordinates of OSM")
    parser_replace.add_argument("-B", "--no-backup", dest="nobackup",
                                action="store_true",
                                help="do NOT backup original OSM")
    parser_replace.add_argument("-c", "--coord-file", dest="coordfile",
                                required=True,
                                help="file of new coordinates")
    parser_replace.add_argument("infile", help="input OSKAR sky model")
    parser_replace.set_defaults(func=cmd_replace)
    #
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
