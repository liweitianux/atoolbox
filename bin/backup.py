#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI
# MIT license
#
# 2017-05-11

"""
Backup the specified files & directories with the metadata (e.g., ownership,
permission) preserved as much as possible.

The extended attributes (i.e., xattr) are only available on Linux only.


Example configuration (YAML format)
-----------------------------------
src_root : /
sources : []
dest_root : /backup
dest_remove : []
syspath : []
-----------------------------------

References
----------
"""

import os
import sys
import stat
import argparse
import subprocess
import logging
from fnmatch import fnmatch
from datetime import datetime

try:
    from yaml import load as yaml_load
except ImportError:
    from ruamel.yaml import safe_load as yaml_load


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
progname = os.path.basename(sys.argv[0])
logger = logging.getLogger(progname)

# Protected system paths
# The destination backup path is checked against these paths, and cannot
# located where matches these paths, thus to avoid the system files being
# overwritten or removed in accidence.
syspath = set(["/", "/home", "/media", "/mnt", "/root",
               "/bin/*", "/boot/*", "/dev/*", "/etc/*", "/lib/*", "/lib64/*",
               "/proc/*", "/sbin/*", "/sys/*", "/usr/*", "/var/*"])


class Backup:
    """
    Backup files with metadata preserved.
    """
    def __init__(self, configfile, dryrun=False, debug=False):
        self.configfile = configfile
        self.config = yaml_load(open(configfile))
        logger.info("Loaded configuration file: %s" % configfile)
        src_root = self.config.get("src_root", "/")
        if os.path.isabs(src_root):
            self.src_root = src_root
            logger.info("Source root directory: %s" % self.src_root)
        else:
            raise ValueError("Source root must be an absolute path")
        self.syspath = syspath.union(self.config.get("syspath", []))
        logger.info("Protected system paths: {0}".format(self.syspath))
        dest_root = os.path.expanduser(self.config["dest_root"])
        logger.info("Check backup destination against protected paths ...")
        self.dest_root = self.check_dest(dest_root)
        logger.info("Backup destination: %s" % self.dest_root)
        self.dryrun = dryrun
        logger.info("Dry run mode: %s" % dryrun)
        self.debug = debug
        logger.info("Show DEBUG information: %s" % debug)

    def check_dest(self, dest):
        """
        Check the given destination backup directory against the protected
        system paths.
        """
        dest = dest.rstrip("/")
        if not os.path.isabs(dest):
            raise ValueError("Destination must be an absolute path")
        for sp in self.syspath:
            if fnmatch(dest, sp):
                raise ValueError("Destination cannot under "
                                 "protected paths: %s" % dest)
        return dest

    def backup(self):
        """
        Do backup sources to the destination.

        The necessary destination directories are created before copying
        files.
        """
        if not os.path.exists(self.dest_root):
            logger.info("Create destination root: " % self.dest_root)
            self.mkdir(self.dest_root, dryrun=self.dryrun)
        for p in self.config["sources"]:
            src = os.path.join(self.src_root, p.lstrip("/"))
            dest = os.path.join(self.dest_root, p.lstrip("/"))
            if os.path.isdir(src):
                src_dir = src
                dest_dir = dest
            else:
                src_dir = os.path.dirname(src)
                dest_dir = os.path.dirname(dest)
            self.mkdirs(dest_dir, ref=src_dir, dryrun=self.dryrun)
            self.copy(src, dest, dryrun=self.dryrun, debug=self.debug)

    def cleanup(self):
        """
        Remove the obsolete files/directories from the destination.
        """
        for p in self.config.get("dest_remove", []):
            path = os.path.join(self.dest_root, p.lstrip("/"))
            self.remove(path, dest_root=self.dest_root,
                        dryrun=self.dryrun, debug=self.debug)

    @staticmethod
    def copy(src, dest, dryrun=False, debug=False):
        """
        Copy file/directory using `rsync` with metadata preserved, and to
        keep directory contents in sync.

        Use `rsync --version` to check the available capabilities.
        """
        if os.path.isdir(src):
            src = src.rstrip("/") + "/"
            dest = dest.rstrip("/") + "/"
        logger.info("Copy & sync: %s -> %s" % (src, dest))
        args = ["--archive", "--hard-links", "--numeric-ids",
                "--delete", "--delete-after"]
        if os.uname().sysname == "Linux":
            args += ["--acls", "--xattrs"]
        if debug:
            args += ["--verbose"]
        cmd = ["rsync"] + args + [src, dest]
        if not dryrun:
            subprocess.check_call(cmd)

    @staticmethod
    def remove(path, dest_root, dryrun=False, debug=False):
        """
        Remove the specified file/directory using `rm -rf`, to clean
        up the destination backup.

        The specified path must locate under the `dest_root` for safety.
        """
        if not fnmatch(path, dest_root+"/*"):
            raise ValueError("Not allowed to remove file/directory "
                             "outside destination: %s" % path)
        if not os.path.exists(path):
            return
        logger.info("Remove: %s" % path)
        args = ["-r", "-f"]
        if debug:
            args += ["-v"]
        cmd = ["rm"] + args + [path]
        if not dryrun:
            subprocess.check_call(cmd)

    @classmethod
    def mkdirs(cls, dest, ref=None, dryrun=False):
        """
        Recursively create the destination directories.
        """
        head, tail = os.path.split(dest)
        head_ref, tail_ref = os.path.split(ref)
        if not tail:
            head, tail = os.path.split(head)
            head_ref, tail_ref = os.path.split(head_ref)
        if head and tail and not os.path.exists(head):
            cls.mkdirs(head, ref=head_ref, dryrun=dryrun)
            if tail == os.path.curdir:
                # `xxx/newdir/.` exists if `xxx/newdir` exists
                return
        cls.mkdir(dest, ref=ref, dryrun=dryrun)

    @classmethod
    def mkdir(cls, dest, ref=None, dryrun=False):
        """
        Make the destination directory with the same metadata w.r.t. the
        reference directory.

        Parameters
        ----------
        dest : str
            The destination directory to be created
        ref : str, optional
            The reference directory, whose metadata will be mirrored to
            the destination directory.
        dryrun : bool
            If True, do not actually create the directory
        """
        if os.path.exists(dest):
            return
        logger.info("Create directory: %s" % dest)
        if not dryrun:
            os.mkdir(dest)
        if ref and not dryrun:
            cls.copystat(ref, dest)

    @classmethod
    def copystat(cls, src, dest, copy_own=True, copy_xattr=True):
        """
        Copy all stat info (mode bits, atime, mtime, flags) from `src` to
        `dest`.  If `copy_own=True`, the uid and gid are also copied.
        If `copy_xattr=True`, the extended attributes are also copied
        (only available on Linux).
        """
        st = os.stat(src)
        mode = stat.S_IMODE(st.st_mode)
        os.chmod(dest, mode=mode)
        os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns))
        if hasattr(st, "st_flags"):
            os.chflags(dest, flags=st.st_flags)
        if copy_own:
            os.chown(dest, uid=st.st_uid, gid=st.st_gid)
        if copy_xattr:
            cls.copyxattr(src, dest)

    @staticmethod
    def copyxattr(src, dest):
        """
        Copy the extended attributes (xattr) from `src` to `dest`.

        NOTE: xattr only available on Linux.
        """
        if not hasattr(os, "listxattr"):
            return
        for name in os.listxattr(src):
            value = os.getxattr(src, name)
            os.setxattr(dest, name, value)


def main():
    parser = argparse.ArgumentParser(
        description="Backup files preserving metadata")
    parser.add_argument("-n", "--dry-run", dest="dryrun", action="store_true",
                        help="dry run, do not perform actual action")
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true",
                        help="be quiet")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="show verbose debug information")
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args()

    if args.quiet and not args.dryrun:
        logging.basicConfig(level=logging.WARNING)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    now = datetime.now()
    logger.info("=== %s @ %s ===" % (" ".join(sys.argv), now.isoformat()))
    if args.dryrun:
        logger.info("*** DRY RUN ***")
    backup = Backup(args.config, dryrun=args.dryrun, debug=args.debug)
    backup.backup()
    backup.cleanup()
    logger.info("=== Backup Finished! @ %s ===" % datetime.now().isoformat())


if __name__ == "__main__":
    main()
