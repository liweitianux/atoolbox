#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# LICENSE:
# ---------------------------------------------------------------------------
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
# ---------------------------------------------------------------------------
#
# Based on Matrin Fiedler's "rebuild_db.py" v1.0-rc1 (2006-04-26):
# http://shuffle-db.sourceforge.net/
#

from __future__ import print_function


__title__   = "iPod Shuffle Database Builder"
__author__  = "Aaron LI"
__version__ = "2.0.2"
__date__    = "2016-04-16"


import sys
import os
import operator
import array
import random
import fnmatch
import operator
import string
import argparse
import functools
import shutil
from collections import OrderedDict


domains = []
total_count = 0


class LogObj:
    """
    Print and log the process information.
    """
    def __init__(self, filename=None):
        self.filename = filename

    def open(self):
        if self.filename:
            try:
                self.logfile = open(self.filename, "w")
            except IOError:
                self.logfile = None
        else:
            self.logfile = None

    def log(self, line="", end="\n"):
        value = line + end
        if self.logfile:
            self.logfile.write(value)
        print(value, end="")

    def close(self):
        if self.logfile:
            self.logfile.close()


class Rule:
    """
    A RuleSet for the way to handle the found playable files.
    """
    SUPPORT_PROPS = ("filename", "size", "ignore", "type",
                     "shuffle", "reuse", "bookmark")

    def __init__(self, conditions=None, actions=None):
        self.conditions = conditions
        self.actions    = actions

    @classmethod
    def parse(cls, rule):
        """
        Parse the whole line of a rule.

        Syntax:
            condition1, condition2, ...: action1, action2, ...

        condition examples:
          * filename ~ "*.mp3"
          * size > 100000
        action examples:
          * ignore = 1
          * shuffle = 1

        Return: a object of this class with the parsed rule.
        """
        conditions, actions = rule.split(":")
        conditions = list(map(cls.parse_condition, conditions.split(",")))
        actions = dict(map(cls.parse_action, actions.split(",")))
        return cls(conditions, actions)

    @classmethod
    def parse_condition(cls, cond):
        sep_pos = min([ cond.find(sep) for sep in "~=<>" \
                        if cond.find(sep)>0 ])
        prop = cond[:sep_pos].strip()
        if prop not in cls.SUPPORT_PROPS:
            raise ValueError("WARNING: unknown property '%s'" % prop)
        return (prop, cond[sep_pos],
                cls.parse_value(cond[sep_pos+1:].strip()))

    @classmethod
    def parse_action(cls, action):
        prop, value = map(str.strip, action.split("=", 1))
        if prop not in cls.SUPPORT_PROPS:
            raise ValueError("WARNING: unknown property '%s'" % prop)
        return (prop, cls.parse_value(value))

    @staticmethod
    def parse_value(value):
        value = value.strip().strip('"').strip("'")
        try:
            return int(value)
        except ValueError:
            return value

    def match(self, props):
        """
        Check whether the given props match all the conditions.
        """
        def match_condition(props, cond):
            """
            Check whether the given props match the given condition.
            """
            try:
                prop, op, ref = props[cond[0]], cond[1], cond[2]
            except KeyError:
                return False
            if op == "~":
                return fnmatch.fnmatchcase(prop.lower(), ref.lower())
            elif op == "=":
                return prop == ref
            elif op == ">":
                return prop > ref
            elif op == "<":
                return prop < ref
            else:
                return False
        #
        return functools.reduce(operator.and_,
                                [ match_condition(props, cond) \
                                        for cond in self.conditions ],
                                True)


class Entries:
    """
    Walk through the directory to find all files, and filter by the
    extensions to get all the playable files.
    """
    PLAYABLE_EXTS = (".mp3", ".m4a", ".m4b", ".m4p", ".aa", ".wav")

    def __init__(self, dirs=[], rename=True, recursive=True, ignore_dup=True):
        self.entries = []
        self.add_dirs(dirs=dirs, rename=rename, recursive=recursive,
                ignore_dup=ignore_dup)

    def add_dirs(self, dirs=[], rename=True, recursive=True, ignore_dup=True):
        for dir in dirs:
            self.add_dir(dir=dir, rename=rename, recursive=recursive,
                    ignore_dup=ignore_dup)

    def add_dir(self, dir, rename=True, recursive=True, ignore_dup=True):
        global logobj
        if recursive:
            # Get all directories, and rename them if needed
            dirs = []
            for dirName, subdirList, fileList in os.walk(dir):
                dirs.append(dirName)
            for dirName in dirs:
                newDirName = self.get_newname(dirName)
                if rename and newDirName != dirName:
                    logobj.log("Rename: '%s' -> '%s'" % (dirName, newDirName))
                    shutil.move(dirName, newDirName)
            # Get all files
            files = []
            for dirName, subdirList, fileList in os.walk(dir):
                files.extend([ os.path.join(dirName, f) for f in fileList ])
        else:
            # rename the directory if needed
            newDir = self.get_newname(dir)
            if rename and newDir != dir:
                logobj.log("Rename: '%s' -> '%s'" % (dir, newDir))
                shutil.move(dir, newDir)
            files = [ os.path.join(newDir, f) for f in self.listfiles(newDir) ]
        #
        for fn in files:
            # rename filename if needed
            newfn = self.get_newname(fn)
            if rename and newfn != fn:
                logobj.log("Rename: '%s' -> '%s'" % (fn, newfn))
                shutil.move(fn, newfn)
                fn = newfn
            # filter by playable extensions
            if os.path.splitext(fn)[1].lower() not in self.PLAYABLE_EXTS:
                continue
            if ignore_dup and (fn in self.entries):
                continue
            self.entries.append(fn)
            print("Entry: %s" % fn)

    @staticmethod
    def listfiles(path, ignore_hidden=True):
        """
        List only files of a directory
        """
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                if ignore_hidden and f[0] != ".":
                    yield f
                else:
                    yield f

    @staticmethod
    def get_newname(path):
        def conv_char(ch):
            safe_char = string.ascii_letters + string.digits + "-_"
            if ch in safe_char:
                return ch
            return "_"
        #
        if path == ".":
            return path
        dirname, basename = os.path.split(path)
        base, ext = os.path.splitext(basename)
        newbase = "".join(map(conv_char, base))
        if basename == newbase+ext:
            return os.path.join(dirname, basename)
        if os.path.exists("%s/%s%s" % (dirname, newbase, ext)):
            i = 0
            while os.path.exists("%s/%s_%d%s" % (dirname, newbase, i, ext)):
                i += 1
            newbase += "_%d" % i
        newname = "%s/%s%s" % (dirname, newbase, ext)
        return newname

    def fix_and_sort(self):
        """
        Fix the entries' pathes (should starts with "/"), and sort.
        """
        self.entries = [ "/"+f.lstrip("./") for f in self.entries ]
        self.entries.sort()

    def apply_rules(self, rules):
        """
        Apply rules to the found entries.
        The filtered/updated entries and properties are saved in:
        'self.entries_dict'
        """
        self.entries_dict = OrderedDict()

        for fn in self.entries:
            # set default properties
            props = {
                "filename": fn,
                "size":     os.stat(fn[1:]).st_size,
                "ignore":   0,
                "type":     1,
                "shuffle":  1,
                "bookmark": 0
            }
            # check and apply rules
            for rule in rules:
                if rule.match(props):
                    props.update(rule.actions)
            #
            if props["ignore"]:
                continue
            #
            self.entries_dict[fn] = props

    def get_entries(self):
        return self.entries_dict.items()


class iTunesSD:
    """
    Class to handle the iPod Shuffle main database
    "iPod_Control/iTunes/iTunesSD"
    """
    def __init__(self, dbfile="./iPod_Control/iTunes/iTunesSD"):
        self.dbfile = dbfile
        self.load()

    def load(self):
        """
        Load original header and entries.
        """
        self.old_entries = {}
        self.header_main = array.array("B")  # unsigned integer array
        self.header_entry = array.array("B")  # unsigned integer array
        db = open(self.dbfile, "rb")
        try:
            self.header_main.fromfile(db, 18)
            self.header_entry.fromfile(db, 33)
            db.seek(18)
            entry = db.read(558)
            while len(entry) == 558:
                filename = entry[33::2].split(b"\0", 1)[0]
                self.old_entries[filename] = entry
                entry = db.read(558)
        except EOFError:
            pass
        db.close()
        print("Loaded %d entries from existing database" % \
                len(self.old_entries))

    def build_header(self, force=False):
        global logobj
        # rebuild database header
        if force or len(self.header_main) != 18:
            logobj.log("Rebuild iTunesSD main header ...")
            del self.header_main[:]
            self.header_main.fromlist([0,0,0,1,6,0,0,0,18] + [0]*9)
        if force or len(self.header_entry) != 33:
            logobj.log("Rebuild iTunesSD entry header ...")
            del self.header_entry[:]
            self.header_entry.fromlist([0,2,46,90,165,1] + [0]*20 + \
                                       [100,0,0,1,0,2,0])

    def add_entries(self, entries, reuse=True):
        """
        Prepare the entries for database
        """
        self.entries = OrderedDict()

        for fn, props in entries.get_entries():
            if reuse and props.get("reuse") and (fn in self.old_entries):
                # retrieve entry from old entries
                entry = self.old_entries[fn]
            else:
                # build new entry
                self.header_entry[29] = props["type"]
                entry_data = "".join([ c+"\0" for c in fn[:261] ]) + \
                        "\0"*(558 - len(self.header_entry) - 2*len(fn))
                entry = self.header_entry.tostring() + \
                        entry_data.encode("utf-8")
            # modify the shuffle and bookmark flags
            entry = entry[:555] + chr(props["shuffle"]).encode("utf-8") + \
                    chr(props["bookmark"]).encode("utf-8") + entry[557]
            #
            self.entries[fn] = entry

    def write(self, dbfile=None):
        if dbfile is None:
            dbfile = self.dbfile
        # Make a backup
        if os.path.exists(dbfile):
            shutil.copy2(dbfile, dbfile+"_bak")

        # write main database file
        with open(dbfile, "wb") as db:
            self.header_main.tofile(db)
            for entry in self.entries.values():
                db.write(entry)
            # Update database header
            num_entries = len(self.entries)
            db.seek(0)
            db.write(b"\0%c%c" % (num_entries>>8, num_entries&0xFF))


class iTunesPState:
    """
    iPod Shuffle playback state database: "iPod_Control/iTunes/iTunesPState"
    """
    def __init__(self, dbfile="iPod_Control/iTunes/iTunesPState"):
        self.dbfile = dbfile
        self.load()

    def load(self):
        with open(self.dbfile, "rb") as db:
            a = array.array("B")
            a.fromstring(db.read())
            self.PState = a.tolist()

    def update(self, volume=None):
        if len(self.PState) != 21:
            # volume 29, FW ver 1.0
            self.PState = self.listval(29) + [0]*15 + self.listval(1)
        # track 0, shuffle mode, start of track
        self.PState[3:15] = [0]*6 + [1] + [0]*5
        if volume is not None:
            self.PState[:3] = self.listval(volume)

    def write(self, dbfile=None):
        if dbfile is None:
            dbfile = self.dbfile
        # Make a backup
        if os.path.exists(dbfile):
            shutil.copy2(dbfile, dbfile+"_bak")

        with open(dbfile, "wb") as db:
            array.array("B", self.PState).tofile(db)

    @staticmethod
    def listval(i):
        if i < 0:
            i += 0x1000000
        return [i&0xFF, (i>>8)&0xFF, (i>>16)&0xFF]


class iTunesStats:
    """
    iPod Shuffle statistics database: "iPod_Control/iTunes/iTunesStats"
    """
    def __init__(self, dbfile="iPod_Control/iTunes/iTunesStats"):
        self.dbfile = dbfile

    def write(self, count, dbfile=None):
        if dbfile is None:
            dbfile = self.dbfile
        # Make a backup
        if os.path.exists(dbfile):
            shutil.copy2(dbfile, dbfile+"_bak")

        with open(dbfile, "wb") as db:
            data = self.stringval(count) + "\0"*3 + \
                     (self.stringval(18) + "\xff"*3 + "\0"*12) * count
            db.write(data.encode("utf-8"))

    @staticmethod
    def stringval(i):
        if i < 0:
            i += 0x1000000
        return "%c%c%c" % (i&0xFF, (i>>8)&0xFF, (i>>16)&0xFF)


class iTunesShuffle:
    """
    iPod shuffle database: "iPod_Control/iTunes/iTunesShuffle"
    """
    def __init__(self, dbfile="iPod_Control/iTunes/iTunesShuffle"):
        self.dbfile = dbfile

    def shuffle(self, entries):
        """
        Generate the shuffle sequences for the entries, and take care
        of the "shuffle" property.
        """
        shuffle_prop = [ props["shuffle"]
                         for fn, props in entries.get_entries() ]
        shuffle_idx = [ idx for idx, s in enumerate(shuffle_prop) if s == 1 ]
        shuffled = shuffle_idx.copy()
        random.seed()
        random.shuffle(shuffled)
        shuffle_seq = list(range(len(shuffle_prop)))
        for i, idx in enumerate(shuffle_idx):
            shuffle_seq[idx] = shuffled[i]
        self.shuffle_seq = shuffle_seq

    def write(self, dbfile=None):
        if dbfile is None:
            dbfile = self.dbfile
        # Make a backup
        if os.path.exists(dbfile):
            shutil.copy2(dbfile, dbfile+"_bak")

        with open(dbfile, "wb") as db:
            data = "".join(map(iTunesStats.stringval, self.shuffle_seq))
            db.write(data.encode("utf-8"))


def main():
    prog_basename = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # command line arguments
    parser = argparse.ArgumentParser(
            description="Rebuild iPod Shuffle Database",
            epilog="Version: %s (%s)\n\n" % (__version__, __date__) + \
                   "Only 1st and 2nd iPod Shuffle supported!\n\n" + \
                   "The script must be placed under the iPod's root directory")
    parser.add_argument("-f", "--force", dest="force", action="store_true",
            help="always rebuild database entries, do NOT reuse old ones")
    parser.add_argument("-M", "--no-rename", dest="norename",
            action="store_false", default=True,
            help="do NOT rename files")
    parser.add_argument("-V", "--volume", dest="volume", type=int,
            help="set playback volume (0 - 38)")
    parser.add_argument("-r", "--rulesfile", dest="rulesfile",
            default="%s.rules" % prog_basename,
            help="additional rules filename")
    parser.add_argument("-l", "--logfile", dest="logfile",
            default="%s.log" % prog_basename,
            help="log output filename")
    parser.add_argument("dirs", nargs="*",
            help="directories to be searched for playable files")
    args = parser.parse_args()

    flag_reuse = not args.force

    # Start logging
    global logobj
    logobj = LogObj(args.logfile)
    logobj.open()

    # Rules for how to handle the found playable files
    rules = []
    # Add default rules
    rules.append(Rule(conditions=[("filename", "~", "*.mp3")],
                      actions={"type":1, "shuffle":1, "bookmark":0}))
    rules.append(Rule(conditions=[("filename", "~", "*.m4?")],
                      actions={"type":2, "shuffle":1, "bookmark":0}))
    rules.append(Rule(conditions=[("filename", "~", "*.m4b")],
                      actions={"shuffle":0, "bookmark":1}))
    rules.append(Rule(conditions=[("filename", "~", "*.aa")],
                      actions={"type":1, "shuffle":0, "bookmark":1, "reuse":1}))
    rules.append(Rule(conditions=[("filename", "~", "*.wav")],
                      actions={"type":4, "shuffle":0, "bookmark":0}))
    rules.append(Rule(conditions=[("filename", "~", "*.book.???")],
                      actions={"shuffle":0, "bookmark":1}))
    rules.append(Rule(conditions=[("filename", "~", "*.announce.???")],
                      actions={"shuffle":0, "bookmark":0}))
    rules.append(Rule(conditions=[("filename", "~", "/backup/*")],
                      actions={"ignore":1}))
    # Load additional rules
    try:
        for line in open(args.rulesfile, "r").readlines():
            rules.append(Rule.parse(line))
        logobj.log("Loaded additional rules from file: %s" % args.rulesfile)
    except IOError:
        pass

    # cd to the directory of this script
    os.chdir(os.path.dirname(sys.argv[0]))

    if not os.path.isdir("iPod_Control/iTunes"):
        logobj.log("ERROR: No iPod control directory found!")
        logobj.log("Please make sure that:")
        logobj.log("(*) this script is placed under the iPod's root directory")
        logobj.log("(*) the iPod was correctly initialized with iTunes")
        sys.exit(1)

    # playable entries
    logobj.log("Search for playable entries ...")
    entries = Entries()
    if args.dirs:
        for dir in args.dirs:
            entries.add_dir(dir=dir, recursive=True, rename=args.norename)
    else:
        entries.add_dir(".", recursive=True, rename=args.norename)
    entries.fix_and_sort()
    logobj.log("Apply rules to entries ...")
    entries.apply_rules(rules=rules)

    # read main database file
    logobj.log("Update main database ...")
    db = iTunesSD(dbfile="iPod_Control/iTunes/iTunesSD")
    db.build_header(force=args.force)
    db.add_entries(entries=entries, reuse=flag_reuse)
    assert len(db.entries) == len(entries.get_entries())
    db.write()
    logobj.log("Added %d entries ..." % len(db.entries))

    # other misc databases
    logobj.log("Update playback state database ...")
    db_pstate = iTunesPState(dbfile="iPod_Control/iTunes/iTunesPState")
    db_pstate.update(volume=args.volume)
    db_pstate.write()
    logobj.log("Update statistics database ...")
    db_stats = iTunesStats(dbfile="iPod_Control/iTunes/iTunesStats")
    db_stats.write(count=len(db.entries))
    logobj.log("Update shuffle database ...")
    db_shuffle = iTunesShuffle(dbfile="iPod_Control/iTunes/iTunesShuffle")
    db_shuffle.shuffle(entries=entries)
    db_shuffle.write()

    logobj.log("The iPod Shuffle database was rebuilt successfully!")

    logobj.close()


if __name__ == "__main__":
    main()

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
