#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI
# MIT license
#
# 2017-04-17
#

"""
Rename email filename according to the given pattern.
"""

import os
import sys
import email
import argparse


class Email:
    def __init__(self, filename):
        self.filename = filename
        self.message = email.message_from_file(open(filename))
        self.date_ = email.utils.parsedate_to_datetime(self.message["Date"])
        self.from_ = email.utils.parseaddr(self.message["From"])[-1]

    @property
    def headers(self):
        return {
            "date": self.date_.isoformat(),
            "from": self.from_
        }

    def rename(self, pattern, dryrun=False):
        newname = pattern.format(**self.headers)
        newfp = os.path.join(os.path.dirname(self.filename), newname)
        print("Rename: '%s' -> '%s'" % (self.filename, newfp), file=sys.stderr)
        if not dryrun:
            os.renames(self.filename, newfp)


def main():
    # format: <filename>:2,<flags>
    default_pattern = "{date}_{from}:2,S"

    parser = argparse.ArgumentParser(
        description="Rename email filename according the given pattern")
    parser.add_argument("-p", "--pattern", dest="pattern",
                        default=default_pattern,
                        help="New filename pattern (default: '%s')" %
                             default_pattern)
    parser.add_argument("-n", "--dry-run", dest="dryrun",
                        action="store_true", help="Dry run")
    parser.add_argument("emails", nargs="+",
                        help="List of email files to be renamed")
    args = parser.parse_args()

    for fn in args.emails:
        eml = Email(fn)
        eml.rename(pattern=args.pattern, dryrun=args.dryrun)


if __name__ == "__main__":
    main()
