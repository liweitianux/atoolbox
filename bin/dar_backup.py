#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI
# MIT license
#
# 2017-03-08

"""
Full/differential backup using 'dar' with 'par2'.

This script accepts a configuration file, which is an simple YAML file
and defines all necessary options for 'dar'.
The 'dar'-created archives are stored under the same directory as the
configuration file, and 'par2' is used to create redundancy data for
possible repairs at the same time, and finally the archives are
tested/verified by 'dar'.


Example configuration file for system/root backup
-------------------------------------------------
# recommended format: <hostname>-root-<distro>-{date}
name: "office-root-opensuse-{date}"
fs_root: "/"
is_system: true
prune: []
exclude: []
exclude_compression: []


Example configuration file for user home backup
-----------------------------------------------
# recommended format: <hostname>-home-<user>-{date}
name: "office-home-aly-{date}"
fs_root: "/home/aly"
prune: [".cache",]
exclude: []
exclude_compression: []


Credit/references
-----------------
[?] http://www.halfgaar.net/backing-up-unix
[?] http://dar.linux.free.fr/doc/man/dar.html
[?] https://github.com/Parchive/par2cmdline
[?] http://dar.linux.free.fr/doc/samples/dar_par_create.duc
[?] http://dar.linux.free.fr/doc/samples/dar_par_test.duc
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from glob import glob

import yaml


progname = os.path.basename(sys.argv[0])
logger = logging.getLogger(progname)


class DarSettings:
    """
    dar settings
    """
    # directory (relative path) for storing the isolated catalogs
    catalog_path = "catalogs"

    # date format for output archive name
    date_fmt = "%Y%m%dT%H%M"

    # Default settings
    args_common = [
        "--min-digits", "3,3,3",
        "--noconf",  # do not try to read /etc/darrc or ~/.darrc
    ]
    args_default = [
        "--alter=atime",  # do not preserve atime
        "--alter=no-case",  # case insensitive mode
        "--alter=glob",  # glob expression mode (instead of regex mode)
        "--alter=binary",  # use 2^10 instead of 10^3
        "--compression=bzip2",
        "--empty-dir",  # create empty directories for the excluded
        "--hash", "sha512",  # calculate the hash of slices on the fly
        "--no-overwrite",
        "--no-mount-points",  # stay in the same filesystem
    ]
    size_slice = "2G"
    exclude = set(["*~", ".*~", ".~*"])
    exclude_compression = set([
        "*.7z", "*.ape", "*.avi", "*.bz2", "*.deb", "*.exe", "*.flac",
        "*.flv", "*.gz", "*.iso", "*.jar", "*.jpg", "*.jpeg",
        "*.m4a", "*.m4v", "*.mkv", "*.mov", "*.mp3", "*.mp4", "*.ogg",
        "*.rar", "*.rpm", "*.tar.bz2", "*.tar.gz", "*.tar.xz",
        "*.tbz", "*.tgz", "*.txz", "*.wmv", "*.xz", "*.zip",
    ])
    prune = set()
    verbose = set()

    # Parchive
    redundancy = 5  # 5% of redundancy for par2
    cmd_par_create = ("par2 create -r{redundancy} -n1 "
                      "'%p/%b.%N.par2' '%p/%b.%N.%e'")
    cmd_par_test = "par2 verify '%p/%b.%N.par2'"

    # Whether it is a backup of system root
    is_system = False
    prune_system = set(["dev/pts", "dev/shm", "home", "lost+found",
                        "media", "mnt", "proc", "run", "sys",
                        "tmp", "var/cache", "var/tmp"])

    def __init__(self, configfile, verbose=False, dry_run=False):
        if verbose:
            self.verbose = set(["treated", "messages"])
        else:
            self.verbose = set()
        if dry_run:
            self.args_common += ["--dry-run"]

        self.path = os.path.dirname(os.path.abspath(configfile))
        settings = yaml.load(open(configfile))
        self.merge_settings(settings)

    def merge_settings(self, settings):
        self.name = settings["name"]
        self.fs_root = settings["fs_root"]
        self.is_system = settings.get("is_system", self.is_system)
        self.date_fmt = settings.get("date_fmt", self.date_fmt)
        self.catalog_path = settings.get("catalog_path", self.catalog_path)
        self.size_slice = settings.get("size_slice", self.size_slice)
        self.redundancy = settings.get("redundancy", self.redundancy)
        self.verbose = self.verbose.union(settings.get("verbose", set()))
        self.prune = self.prune.union(settings.get("prune", set()))
        self.exclude = self.exclude.union(settings.get("exclude", set()))
        self.exclude_compression = self.exclude_compression.union(
            settings.get("exclude_compression", set()))

    def archive_name(self, date=None):
        # Return the archive name with date substituted
        if date is None:
            date = datetime.now().strftime(self.date_fmt)
        name = os.path.join(self.path, self.name.format(date=date))
        return name

    def last_archive_name(self):
        # Get the last archive (backup data or isolated catalog)
        # used as the reference archive for a differential backup.
        name = self.last_backup()
        if name is None:
            name = self.last_catalog()
        return name

    def last_backup(self):
        # Return the name of last backup if found, otherwise, return None
        backup_glob = self.archive_name(date="*") + ".dar"
        try:
            backup = sorted(glob(backup_glob))[-1]
            name = os.path.splitext(os.path.splitext(backup)[0])[0]
            logger.info("Found last backup: {0}".format(name))
            return name
        except IndexError:
            logger.info("No existing backups found!")
            return None

    def last_catalog(self):
        # Return the name of last isolated catalog if found, otherwise,
        # return None
        basename = os.path.basename(self.archive_name(date="*")) + ".dar"
        catalog_glob = os.path.join(self.path, self.catalog_path, basename)
        try:
            catalog = sorted(glob(catalog_glob))[-1]
            name = os.path.splitext(os.path.splitext(catalog)[0])[0]
            logger.info("Found last catalog: {0}".format(name))
            return name
        except IndexError:
            logger.info("No existing catalogs found!")
            return None

    @property
    def args_create(self):
        cmd_par = self.cmd_par_create.format(redundancy=self.redundancy)
        args = self.args_default + [
            "--execute", cmd_par,
            "--fs-root", self.fs_root,
            "--slice", self.size_slice,
        ]
        args += ["--verbose=%s" % item for item in self.verbose]
        for item in self.exclude:
            args += ["--exclude", item]
        for item in self.exclude_compression:
            args += ["-Z", item]
        for item in self.prune:
            args += ["--prune", item]
        if self.is_system:
            for item in self.prune_system:
                args += ["--prune", item]
        logger.info("args_create: {0}".format(args))
        return args

    @property
    def args_test(self):
        args = ["--execute", self.cmd_par_test]
        args += ["--verbose=%s" % item for item in self.verbose]
        logger.info("args_test: {0}".format(args))
        return args


class DarBackup:
    """
    dar backup with settings
    """
    def __init__(self, settings):
        self.settings = settings
        self.is_system = settings.is_system
        self.path = settings.path
        self.catalog_path = settings.catalog_path
        self.archive_name = settings.archive_name()
        self.last_archive_name = settings.last_archive_name()
        self.args_common = settings.args_common
        self.args_create = settings.args_create
        self.args_test = settings.args_test

    def run(self, dry_run=False):
        if self.is_system and (not self.is_running_as_root()):
            raise RuntimeError("Running as root to backup the system root")

        logger.info("Output archive name: {0}".format(self.archive_name))
        if self.last_archive_name is not None:
            self.backup_diff()
        else:
            self.backup_full()

        if not dry_run:
            self.test_backup()
            self.isolate_catalog()

    def backup_full(self):
        logger.info("Start full backup ...")
        cmd = [
            "dar",
            "--create", self.archive_name,
        ] + self.args_common + self.args_create
        logger.info("Command: {0}".format(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Full backup: DONE!")

    def backup_diff(self):
        logger.info("Start differential backup ...")
        cmd = [
            "dar",
            "--create", self.archive_name,
            "--ref", self.last_archive_name,
        ] + self.args_common + self.args_create
        logger.info("Command: {0}".format(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Differential backup: DONE!")

    def test_backup(self):
        logger.info("Test created backup ...")
        cmd = [
            "dar",
            "--test", self.archive_name,
        ] + self.args_common + self.args_test
        logger.info("Command: {0}".format(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Test backup: DONE!")

    def isolate_catalog(self):
        logger.info("Isolate the catalog from backup ...")
        catalog_dir = os.path.join(self.path, self.catalog_path)
        if not os.path.exists(catalog_dir):
            os.mkdir(catalog_dir)
        catalog = os.path.join(catalog_dir,
                               os.path.basename(self.archive_name))
        logger.info("Output catalog: {0}".format(catalog))
        cmd = [
            "dar",
            "--isolate", catalog,
            "--ref", self.archive_name,
        ] + self.args_common
        logger.info("Command: {0}".format(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Isolate backup catalog: DONE!")

    @staticmethod
    def is_running_as_root():
        return os.getuid() == 0


def main():
    parser = argparse.ArgumentParser(
        description="Backup system/data using dar and par2")
    parser.add_argument("-c", "--config", dest="config", required=True,
                        help="configuration file for dar and archive. " +
                        "NOTE: the backup archive will be placed under " +
                        "the same directory as this configuration file")
    parser.add_argument("-n", "--dry-run", dest="dry_run", action="store_true",
                        help="dry run, do not perform any action")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="show verbose information")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    settings = DarSettings(args.config, verbose=args.verbose,
                           dry_run=args.dry_run)
    dar = DarBackup(settings)
    dar.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
