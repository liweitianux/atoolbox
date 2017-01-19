#!/bin/usr/env python3
# -*- coding: utf-8 -*-
#
# Credits:
# [1] Gaute Hope: gauteh/abunchoftags
#     https://github.com/gauteh/abunchoftags/blob/master/keywsync.cc
#
# TODO:
# * Support case-insensitive tags merge
#   (ref: http://stackoverflow.com/a/1480230)
# * Accept a specified mtime, and only deal with files with newer mtime.
#
# Aaron LI
# Created: 2016-01-24
#

"""
Sync message 'X-Keywords' header with notmuch tags.

* tags-to-keywords:
  Check if the messages in the query have a matching 'X-Keywords' header
  to the list of notmuch tags.
  If not, update the 'X-Keywords' and re-write the message.

* keywords-to-tags:
  Check if the messages in the query have matching notmuch tags to the
  'X-Keywords' header.
  If not, update the tags in the notmuch database.

* merge-keywords-tags:
  Merge the 'X-Keywords' labels and notmuch tags, and update both.
"""

__version__ = "0.1.2"
__date__    = "2016-01-25"

import os
import sys
import argparse
import email

# Require Python 3.4, or install package 'enum34'
from enum import Enum

from notmuch import Database, Query

from imapUTF7 import imapUTF7Decode, imapUTF7Encode


class SyncDirection(Enum):
    """
    Synchronization direction
    """
    MERGE_KEYWORDS_TAGS = 0  # Merge 'X-Keywords' and notmuch tags and
                             # update both
    KEYWORDS_TO_TAGS    = 1  # Sync 'X-Keywords' header to notmuch tags
    TAGS_TO_KEYWORDS    = 2  # Sync notmuch tags to 'X-Keywords' header

class SyncMode(Enum):
    """
    Sync mode
    """
    ADD_REMOVE  = 0  # Allow add & remove tags/keywords
    ADD_ONLY    = 1  # Only allow add tags/keywords
    REMOVE_ONLY = 2  # Only allow remove tags/keywords


class KwMessage:
    """
    Message class to deal with 'X-Keywords' header synchronization
    with notmuch tags.

    NOTE:
    * The same message may have multiple files with different keywords
      (e.g, the same message exported under each label by Gmail)
      managed by OfflineIMAP.
      For example: a message file in OfflineIMAP synced folder of
      '[Gmail]/All Mail' have keywords ['google', 'test']; however,
      the file in synced folder 'test' of the same message only have
      keywords ['google'] without the keyword 'test'.
    * All files associated to the same message are regarded as the same.
      The keywords are extracted from all files and merged.
      And the same updated keywords are written back to all files, which
      results all files finally having the same 'X-Keywords' header.
    * You may only sync the '[Gmail]/All Mail' folder without other
      folders exported according the labels by Gmail.
    """
    # Replace some special characters before mapping keyword to tag
    enable_replace_chars = True
    chars_replace = {
            '/' : '.',
    }
    # Mapping between (Gmail) keywords and notmuch tags (before ignoring tags)
    keywords_mapping = {
            '\\Inbox'     : 'inbox',
            '\\Important' : 'important',
            '\\Starred'   : 'flagged',
            '\\Sent'      : 'sent',
            '\\Muted'     : 'killed',
            '\\Draft'     : 'draft',
            '\\Trash'     : 'deleted',
            '\\Junk'      : 'spam',
    }
    # Tags ignored from syncing
    # These tags are either internal tags or tags handled by maildir flags.
    enable_ignore_tags = True
    tags_ignored = set([
            'new', 'unread', 'attachment', 'signed', 'encrypted',
            'flagged', 'replied', 'passed', 'draft',
    ])
    # Ignore case when merging tags
    tags_ignorecase = True

    # Whether the tags updated against the message 'X-Keywords' header
    tags_updated = False
    # Added & removed tags for notmuch database against 'X-Keywords'
    tags_added   = []
    tags_removed = []
    # Newly updated/merged notmuch tags against 'X-Keywords'
    tags_new     = []

    # Whether the keywords updated against the notmuch tags
    keywords_updated = False
    # Added & removed tags for 'X-Keywords' against notmuch database
    tags_kw_added    = []
    tags_kw_removed  = []
    # Newly updated/merged tags for 'X-Keywords' against notmuch database
    tags_kw_new      = []

    def __init__(self, msg, filename=None):
        self.message  = msg
        self.filename = filename
        self.allfiles = [ fn for fn in msg.get_filenames() ]
        self.tags     = set(msg.get_tags())

    def sync(self, direction, mode=SyncMode.ADD_REMOVE,
             dryrun=False, verbose=False):
        """
        Wrapper function to sync between 'X-Keywords' and notmuch tags.
        """
        if direction == SyncDirection.KEYWORDS_TO_TAGS:
            self.sync_keywords_to_tags(sync_mode=mode, dryrun=dryrun,
                                       verbose=verbose)
        elif direction == SyncDirection.TAGS_TO_KEYWORDS:
            self.sync_tags_to_keywords(sync_mode=mode, dryrun=dryrun,
                                       verbose=verbose)
        elif direction == SyncDirection.MERGE_KEYWORDS_TAGS:
            self.merge_keywords_tags(sync_mode=mode, dryrun=dryrun,
                                     verbose=verbose)
        else:
            raise ValueError("Invalid sync direction: %s" % direction)

    def sync_keywords_to_tags(self, sync_mode=SyncMode.ADD_REMOVE,
                              dryrun=False, verbose=False):
        """
        Wrapper function to sync 'X-Keywords' to notmuch tags.
        """
        self.get_keywords()
        self.map_keywords()
        self.merge_tags(sync_direction=SyncDirection.KEYWORDS_TO_TAGS,
                        sync_mode=sync_mode)
        if dryrun or verbose:
            print('* MSG: %s' % self.message)
            print('  TAG: [%s]  +[%s] -[%s]  =>  [%s]' % (
                ','.join(self.tags), ','.join(self.tags_added),
                ','.join(self.tags_removed), ','.join(self.tags_new)))
        if not dryrun:
            self.update_tags()

    def sync_tags_to_keywords(self, sync_mode=SyncMode.ADD_REMOVE,
                              dryrun=False, verbose=False):
        """
        Wrapper function to sync notmuch tags to 'X-Keywords'
        """
        self.get_keywords()
        self.map_keywords()
        self.merge_tags(sync_direction=SyncDirection.TAGS_TO_KEYWORDS,
                        sync_mode=sync_mode)
        keywords_new = self.map_tags(tags=self.tags_kw_new)
        if dryrun or verbose:
            print('* MSG: %s' % self.message)
            print('* FILES: %s' % ' ; '.join(self.allfiles))
            print('  XKW: {%s}  +[%s] -[%s]  =>  {%s}' % (
                ','.join(self.keywords), ','.join(self.tags_kw_added),
                ','.join(self.tags_kw_removed), ','.join(keywords_new)))
        if not dryrun:
            self.update_keywords(keywords_new=keywords_new)

    def merge_keywords_tags(self, sync_mode=SyncMode.ADD_REMOVE,
                            dryrun=False, verbose=False):
        """
        Wrapper function to merge 'X-Keywords' and notmuch tags
        """
        self.get_keywords()
        self.map_keywords()
        self.merge_tags(sync_direction=SyncDirection.MERGE_KEYWORDS_TAGS,
                        sync_mode=sync_mode)
        keywords_new = self.map_tags(tags=self.tags_kw_new)
        if dryrun or verbose:
            print('* MSG: %s' % self.message)
            print('* FILES: %s' % ' ; '.join(self.allfiles))
            print('  TAG: [%s]  +[%s] -[%s]  =>  [%s]' % (
                ','.join(self.tags), ','.join(self.tags_added),
                ','.join(self.tags_removed), ','.join(self.tags_new)))
            print('  XKW: {%s}  +[%s] -[%s]  =>  {%s}' % (
                ','.join(self.keywords), ','.join(self.tags_kw_added),
                ','.join(self.tags_kw_removed), ','.join(keywords_new)))
        if not dryrun:
            self.update_tags()
            self.update_keywords(keywords_new=keywords_new)

    def get_keywords(self):
        """
        Get 'X-Keywords' header from all files associated with the same
        message, decode, split and merge.

        NOTE: Do NOT simply use the `message.get_header()` method, which
              cannot get the complete keywords from all files.
        """
        keywords_utf7 = []
        for fn in self.allfiles:
            msg = email.message_from_file(open(fn, 'r'))
            val = msg['X-Keywords']
            if val:
                keywords_utf7.append(val)
            else:
                print("WARNING: 'X-Keywords' header not found or empty " +\
                        "for file: %s" % fn, file=sys.stderr)
        keywords_utf7 = ','.join(keywords_utf7)
        if keywords_utf7 != '':
            keywords = imapUTF7Decode(keywords_utf7.encode()).split(',')
            keywords = [ kw.strip() for kw in keywords ]
            # Remove duplications
            keywords = set(keywords)
        else:
            keywords = set()
        self.keywords = keywords
        return keywords

    def map_keywords(self, keywords=None):
        """
        Map keywords to notmuch tags according to the mapping table.
        """
        if keywords is None:
            keywords = self.keywords
        if self.enable_replace_chars:
            # Replace specified characters in keywords
            trans = str.maketrans(self.chars_replace)
            keywords = [ kw.translate(trans) for kw in keywords ]
        # Map keywords to tags
        tags = set([ self.keywords_mapping.get(kw, kw) for kw in keywords ])
        self.tags_kw = tags
        return tags

    def map_tags(self, tags=None):
        """
        Map tags to keywords according to the inversed mapping table.
        """
        if tags is None:
            tags = self.tags
        if self.enable_replace_chars:
            # Inversely replace specified characters in tags
            chars_replace_inv = { v: k for k, v in self.chars_replace.items() }
            trans = str.maketrans(chars_replace_inv)
            tags = [ tag.translate(trans) for tag in tags ]
        # Map keywords to tags
        keywords_mapping_inv = { v:k for k,v in self.keywords_mapping.items() }
        keywords = set([ keywords_mapping_inv.get(tag, tag) for tag in tags ])
        self.keywords_tags = keywords
        return keywords

    def merge_tags(self, sync_direction, sync_mode=SyncMode.ADD_REMOVE,
                   tags_nm=None, tags_kw=None):
        """
        Merge the tags from notmuch database and 'X-Keywords' header,
        according to the specified sync direction and operation restriction.

        TODO: support case-insensitive set operations
        """
        # Added & removed tags for notmuch database against 'X-Keywords'
        tags_added       = []
        tags_removed     = []
        # Newly updated/merged notmuch tags against 'X-Keywords'
        tags_new         = []
        # Added & removed tags for 'X-Keywords' against notmuch database
        tags_kw_added    = []
        tags_kw_removed  = []
        # Newly updated/merged tags for 'X-Keywords' against notmuch database
        tags_kw_new      = []
        #
        if tags_nm is None:
            tags_nm = self.tags
        if tags_kw is None:
            tags_kw = self.tags_kw
        if self.enable_ignore_tags:
            # Remove ignored tags before merge
            tags_nm2 = tags_nm.difference(self.tags_ignored)
            tags_kw2 = tags_kw.difference(self.tags_ignored)
        else:
            tags_nm2 = tags_nm
            tags_kw2 = tags_kw
        #
        if sync_direction == SyncDirection.KEYWORDS_TO_TAGS:
            # Sync 'X-Keywords' to notmuch tags
            tags_added      = tags_kw2.difference(tags_nm2)
            tags_removed    = tags_nm2.difference(tags_kw2)
        elif sync_direction == SyncDirection.TAGS_TO_KEYWORDS:
            # Sync notmuch tags to 'X-Keywords'
            tags_kw_added   = tags_nm2.difference(tags_kw2)
            tags_kw_removed = tags_kw2.difference(tags_nm2)
        elif sync_direction == SyncDirection.MERGE_KEYWORDS_TAGS:
            # Merge both notmuch tags and 'X-Keywords'
            tags_merged     = tags_nm2.union(tags_kw2)
            # notmuch tags
            tags_added      = tags_merged.difference(tags_nm2)
            tags_removed    = tags_nm2.difference(tags_merged)
            # tags for 'X-Keywords'
            tags_kw_added   = tags_merged.difference(tags_kw2)
            tags_kw_removed = tags_kw2.difference(tags_merged)
        else:
            raise ValueError("Invalid synchronization direction")
        # Apply sync operation restriction
        self.tags_added       = []
        self.tags_removed     = []
        self.tags_kw_added    = []
        self.tags_kw_removed  = []
        tags_new              = tags_nm  # Use un-ignored notmuch tags
        tags_kw_new           = tags_kw  # Use un-ignored 'X-Keywords' tags
        if sync_mode != SyncMode.REMOVE_ONLY:
            self.tags_added      = tags_added
            self.tags_kw_added   = tags_kw_added
            tags_new             = tags_new.union(tags_added)
            tags_kw_new          = tags_kw_new.union(tags_kw_added)
        if sync_mode != SyncMode.ADD_ONLY:
            self.tags_removed    = tags_removed
            self.tags_kw_removed = tags_kw_removed
            tags_new             = tags_new.difference(tags_removed)
            tags_kw_new          = tags_kw_new.difference(tags_kw_removed)
        #
        self.tags_new         = tags_new
        self.tags_kw_new      = tags_kw_new
        if self.tags_added or self.tags_removed:
            self.tags_updated = True
        if self.tags_kw_added or self.tags_kw_removed:
            self.keywords_updated = True
        #
        return {
                'tags_updated'     : self.tags_updated,
                'tags_added'       : self.tags_added,
                'tags_removed'     : self.tags_removed,
                'tags_new'         : self.tags_new,
                'keywords_updated' : self.keywords_updated,
                'tags_kw_added'    : self.tags_kw_added,
                'tags_kw_removed'  : self.tags_kw_removed,
                'tags_kw_new'      : self.tags_kw_new,
        }

    def update_keywords(self, keywords_new=None, outfile=None):
        """
        Encode the keywords (default: self.keywords_new) and write back to
        all message files.

        If parameter 'outfile' specified, then write the updated message
        to that file instead of overwriting.

        NOTE:
        * The modification time of the message file should be kept to prevent
          OfflineIMAP from treating it as a new one (and the previous a
          deleted one).
        * All files associated with the same message are updated to have
          the same 'X-Keywords' header.
        """
        if not self.keywords_updated:
            # keywords NOT updated, just skip
            return

        if keywords_new is None:
            keywords_new = self.keywords_new
        #
        if outfile is not None:
            infile  = self.allfiles[0:1]
            outfile = [ os.path.expanduser(outfile) ]
        else:
            infile  = self.allfiles
            outfile = self.allfiles
        #
        for ifname, ofname in zip(infile, outfile):
            msg   = email.message_from_file(open(ifname, 'r'))
            fstat = os.stat(ifname)
            if keywords_new == []:
                # Delete 'X-Keywords' header
                print("WARNING: delete 'X-Keywords' header from file: %s" %
                        ifname, file=sys.stderr)
                del msg['X-Keywords']
            else:
                # Update 'X-Keywords' header
                keywords = ','.join(keywords_new)
                keywords_utf7 = imapUTF7Encode(keywords).decode()
                # Delete then add, to avoid multiple occurrences
                del msg['X-Keywords']
                msg['X-Keywords'] = keywords_utf7
            # Write updated message
            with open(ofname, 'w') as fp:
                fp.write(msg.as_string())
            # Reset the timestamps
            os.utime(ofname, ns=(fstat.st_atime_ns, fstat.st_mtime_ns))

    def update_tags(self, tags_added=None, tags_removed=None):
        """
        Update notmuch tags according to keywords.
        """
        if not self.tags_updated:
            # tags NOT updated, just skip
            return

        if tags_added is None:
            tags_added   = self.tags_added
        if tags_removed is None:
            tags_removed = self.tags_removed
        # Use freeze/thaw for safer transactions to change tag values.
        self.message.freeze()
        for tag in tags_added:
            self.message.add_tag(tag, sync_maildir_flags=False)
        for tag in tags_removed:
            self.message.remove_tag(tag, sync_maildir_flags=False)
        self.message.thaw()


def get_notmuch_revision(dbpath=None):
    """
    Get the current revision and UUID of notmuch database.
    """
    import subprocess
    import tempfile
    if dbpath:
        tf = tempfile.NamedTemporaryFile()
        # Create a minimal notmuch config for the specified dbpath
        config = '[database]\npath=%s\n' % os.path.expanduser(dbpath)
        tf.file.write(config.encode())
        tf.file.flush()
        cmd = 'notmuch --config=%s count --lastmod' % tf.name
        output = subprocess.check_output(cmd, shell=True)
        tf.close()
    else:
        cmd = 'notmuch count --lastmod'
        output = subprocess.check_output(cmd, shell=True)
    # Extract output
    dbinfo = output.decode().split()
    return { 'revision': int(dbinfo[2]), 'uuid': dbinfo[1] }


def main():
    parser = argparse.ArgumentParser(
            description="Sync message 'X-Keywords' header with notmuch tags.")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "v%s (%s)" % (__version__, __date__))
    parser.add_argument("-q", "--query", dest="query", required=True,
            help="notmuch database query string")
    parser.add_argument("-p", "--db-path", dest="dbpath",
            help="notmuch database path (default to try user configuration)")
    parser.add_argument("-n", "--dry-run", dest="dryrun",
            action="store_true", help="dry run")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    # Exclusive argument group for sync mode
    exgroup1 = parser.add_mutually_exclusive_group(required=True)
    exgroup1.add_argument("-m", "--merge-keywords-tags",
            dest="direction_merge", action="store_true",
            help="merge 'X-Keywords' and tags and update both")
    exgroup1.add_argument("-k", "--keywords-to-tags",
            dest="direction_keywords2tags", action="store_true",
            help="sync 'X-Keywords' to notmuch tags")
    exgroup1.add_argument("-t", "--tags-to-keywords",
            dest="direction_tags2keywords", action="store_true",
            help="sync notmuch tags to 'X-Keywords'")
    # Exclusive argument group for tag operation mode
    exgroup2 = parser.add_mutually_exclusive_group(required=False)
    exgroup2.add_argument("-a", "--add-only", dest="mode_addonly",
            action="store_true", help="only add notmuch tags")
    exgroup2.add_argument("-r", "--remove-only", dest="mode_removeonly",
            action="store_true", help="only remove notmuch tags")
    # Parse
    args = parser.parse_args()
    # Sync direction
    if args.direction_merge:
        sync_direction = SyncDirection.MERGE_KEYWORDS_TAGS
    elif args.direction_keywords2tags:
        sync_direction = SyncDirection.KEYWORDS_TO_TAGS
    elif args.direction_tags2keywords:
        sync_direction = SyncDirection.TAGS_TO_KEYWORDS
    else:
        raise ValueError("Invalid synchronization direction")
    # Sync mode
    if args.mode_addonly:
        sync_mode = SyncMode.ADD_ONLY
    elif args.mode_removeonly:
        sync_mode = SyncMode.REMOVE_ONLY
    else:
        sync_mode = SyncMode.ADD_REMOVE
    #
    if args.dbpath:
        dbpath = os.path.abspath(os.path.expanduser(args.dbpath))
    else:
        dbpath = None
    #
    db = Database(path=dbpath, create=False, mode=Database.MODE.READ_WRITE)
    dbinfo = get_notmuch_revision(dbpath=dbpath)
    q = Query(db, args.query)
    total_msgs = q.count_messages()
    msgs = q.search_messages()
    #
    if args.verbose:
        print("# Notmuch database path: %s" % dbpath)
        print("# Database revision: %d (uuid: %s)" %
                (dbinfo['revision'], dbinfo['uuid']))
        print("# Query: %s" % args.query)
        print("# Sync direction: %s" % sync_direction.name)
        print("# Sync mode: %s" % sync_mode.name)
        print("# Total messages to check: %d" % total_msgs)
        print("# Dryn run: %s" % args.dryrun)
    #
    for msg in msgs:
        kwmsg = KwMessage(msg)
        kwmsg.sync(direction=sync_direction, mode=sync_mode,
                   dryrun=args.dryrun, verbose=args.verbose)
    #
    db.close()


if __name__ == "__main__":
    main()

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
