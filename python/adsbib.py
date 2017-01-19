#!/usr/bin/env python3
#
# Copyright (c) 2016 Aaron LI
# MIT License
#

"""
Build and maintain the BibTeX database for a literature of PDF files.

This tool try to extract the metadata from the PDF file, then query
the SAO/NASA Astrophysics Data System (ADS) for details, and finally
build the BibTeX entries.


References/Credits:
[1] SAO/NASA Astrophysics Data System
    https://ui.adsabs.harvard.edu/
[2] ADS API Description
    https://github.com/adsabs/adsabs-dev-api
[3] ADS Search Syntax
    https://adsabs.github.io/help/search/search-syntax
[4] PDFtk: https://www.pdflabs.com/docs/pdftk-man-page/
[5] Poppler utilities: pdftotext
[6] Pybtex: https://pybtex.org/
[7] Requests: HTTP for Humans
    https://github.com/kennethreitz/requests
"""


import os
import re
import json
import fnmatch
import argparse
import logging
import subprocess
from collections import OrderedDict

import requests
import pybtex.database
from pybtex.database import Entry, Person
from unidecode import unidecode


logger = logging.getLogger("adsbib")


class AdsRecord:
    """
    Record data retrieved from ADS service.
    """
    # Regular expression to match the "bibstem" of a bibcode
    BIBSTEM_RE = re.compile(r"^\d{4}([a-z&]+)[\d.]+[a-z.\d]+$", re.IGNORECASE)
    # ADS URL pattern to record
    ADS_URL = "http://adsabs.harvard.edu/abs/%(bibcode)s"

    def __init__(self, data):
        self.data = data
        # Supplement "bibstem" if missing
        self.data["bibstem"] = self.get_bibstem()
        self.bibcode = data["bibcode"]
        self.keys = sorted(data.keys())

    def __str__(self):
        return "<{bibcode} with {keys}>".format(
            bibcode=self.bibcode, keys=",".join(self.keys))

    def pprint(self, indent=4):
        """
        Pretty print the record data
        """
        for key in self.keys:
            print("{indent}{key}: {value}".format(
                indent=" "*indent, key=key, value=self.data[key]))

    def get_bibstem(self):
        """
        Get the "bibstem" from the bibcode, which is the abbreviated name
        of the journal/publication (e.g., ApJ, A&A)
        """
        if "bibstem" in self.data.keys():
            return self.data["bibstem"]
        bibcode = self.data["bibcode"]
        bibstem = self.BIBSTEM_RE.search(bibcode).group(1)
        return bibstem

    def to_bib_data(self):
        """
        Convert to the BibTeX data
        """
        bibtype = self.data["doctype"]
        persons = OrderedDict([
            ("author", [Person(p) for p in self.data["author"]]),
        ])
        fields = OrderedDict([
            ("title", self.data["title"][0]),
            ("adsbibcode", self.data["bibcode"]),
            ("url", self.ADS_URL % {"bibcode": self.data["bibcode"]}),
            ("journal", self.data["pub"]),
            ("year", self.data["year"]),
        ])
        if self.data["bibstem"].lower() == "arxiv":
            fields.update([
                ("archiveprefix", "arXiv"),
                ("eprint", self.data["identifier"][0]),
            ])
        else:
            fields.update([
                ("volume", self.data["volume"]),
                ("page", self.data["page"][0]),
            ])
        if self.data.get("doi"):
            fields["doi"] = self.data["doi"][0]
        if self.data.get("keyword"):
            fields["keywords"] = ", ".join(self.data["keyword"])
        if self.data.get("abstract"):
            fields["abstract"] = self.data["abstract"]
        #
        return (bibtype, persons, fields)


class AdsQuery:
    """
    SAO/NASA ADS query and access class
    """
    # ADS API URLs
    API_URL = "https://api.adsabs.harvard.edu/v1"
    QUERY_URL = API_URL + "/search/query"

    # List of fields to return by the query
    fields = set([
        "abstract",  # abstract of the record
        "ack",  # acknowledgments
        "aff",  # array of the author's affiliations
        "alternate_bibcode",  # list of alternate bibcodes
        "arxiv",  # arXiv ID
        "arxiv_class",  # arXiv class the pre-print was submitted to
        "author",  # array of the author names
        "bibcode",  # canonical ADS bibcode ID
        "bibgroup",  # bibliographic groups the bibcode been associated with
        "bibstem",  # abbreviated name of the journal/publication (e.g., ApJ)
        "copyright",  # copyright
        "data",  # list of sources that have data related
        "database",  # which database the record is associated with
        "doi",  # digital object identifier (DOI)
        "doctype",  # document type, extension to the BibTeX entry types
        "first_author",  # first author of the article
        "grant",  # list of grant IDs and agencies noted
        "id",  # a *non-persistent* ID, for fast look-up of a document
        "identifier",  # array of alternative IDs for the record
        "issue",  # issue number
        "keyword",  # array of normalized and un-normalized keyword values
        "orcid",  # ORCiD IDs supplied by publishers
        "page",  # starting page
        "property",  # array of miscellaneous flags associated
        "pub",  # canonical name of the publication
        "pubdate",  # publication date in the form "YYYY-MM-DD"
        "title",  # title of the record
        "volume",  # volume number
        "year",  # year the record was published
    ])

    def __init__(self, tokenfile):
        self.token = open(tokenfile).read().strip()
        logger.info("Read in ADS API token from: {0}".format(tokenfile))

    def query(self, q, **params):
        headers = {
            "Authorization": "Bearer:{0}".format(self.token),
        }
        params["q"] = q
        if "fl" not in params:
            params["fl"] = ",".join(self.fields)
        r = requests.get(self.QUERY_URL, params=params, headers=headers)
        r.raise_for_status()
        response = json.loads(r.text)
        logger.debug("ADS response header:\n{0}".format(
            response["responseHeader"]))
        logger.debug("ADS response body:\n{0}".format(response["response"]))
        records = [AdsRecord(data) for data in response["response"]["docs"]]
        logger.info("ADS found {0} records".format(len(records)))
        return records


class PDF:
    """
    Class to deal with the PDF file
    """
    # Regular expression to match a DOI
    DOI_RE = re.compile(r"\b(?:doi[:\s]*|)(10\.\d{4,9}/[^\s\|\]\}\?\,\'\"]+)",
                        re.IGNORECASE)
    # Regular expression to match a ADS Bibcode
    BIBCODE_RE = re.compile(r"\b(\d{4}[a-zA-Z0-9&.]{14}[A-Z])")
    # Regular expression to match a arXiv ID
    ARXIV_RE = re.compile(r"\b(?:arxiv:?\s*)([a-z/-]*\d{4}\.?\d{3,5}v?\d+)",
                          re.IGNORECASE)
    # Flag indicating whether the file/metadata is modified
    modified = False

    def __init__(self, filepath):
        self.filepath = filepath
        #
        self.text_metadata = subprocess.check_output(args=[
            "pdftk", self.filepath, "dump_data_utf8"
        ]).decode("utf-8")
        self.text_firstpage = subprocess.check_output(args=[
            "pdftotext", "-f", "1", "-l", "1", self.filepath, "-"
        ]).decode("utf-8")
        #
        self.doi = self.extract(self.DOI_RE)
        self.bibcode = self.extract(self.BIBCODE_RE)
        self.arxiv = self.extract(self.ARXIV_RE)

    @staticmethod
    def is_annotated(path):
        return path.endswith(".annot.pdf")

    @staticmethod
    def is_annotated_of(path1, path2):
        """
        Test whether PDF of path1 is the annotated version of PDF of path2?
        """
        basepath1 = os.path.splitext(path1)[0]
        basepath2 = os.path.splitext(path2)[0]
        return (basepath1 == basepath2 + ".annot")

    @property
    def annotated(self):
        return self.is_annotated(self.filepath)

    def extract(self, regex, grpnum=1):
        value = None
        for text in [self.text_metadata, self.text_firstpage]:
            match = regex.search(text)
            if match:
                value = match.group(grpnum)
                break
        return value

    def add_metadata(self, key, value, overwrite=False):
        metadata = self.text_metadata
        target = "InfoKey: {0}".format(key)
        idx0 = metadata.upper().find(target.upper())
        if idx0 >= 0:
            # Key already exists
            if overwrite:
                # Update existing value
                idx1 = metadata.find("\n", idx0) + 1  # InfoValue's head
                idx2 = metadata.find("\n", idx1)  # InfoValue's tail
                infoval_new = "InfoValue: {0}".format(value)
                self.text_metadata = (metadata[:idx1] + infoval_new +
                                      metadata[idx2:])
                self.modified = True
                logger.info("Updated metadata: {0} = {1}".format(key, value))
            else:
                logger.warning("PDF metadata: key already exists!")
        else:
            # Append the new key
            idx0 = metadata.upper().rfind("InfoValue:".upper())
            idx1 = metadata.find("\n", idx0) + 1
            keyval_new = "".join([
                "InfoBegin\n",
                "InfoKey: {0}\n".format(key),
                "InfoValue: {0}\n".format(value),
            ])
            self.text_metadata = metadata[:idx1] + keyval_new + metadata[idx1:]
            self.modified = True
            logger.info("Added new metadata: {0} = {1}".format(key, value))

    def write(self, outfile=None, overwrite=False):
        """
        Write PDF with updated metadata.
        """
        if not self.modified:
            logger.info("PDF not modified; skip")
            return
        #
        infile = self.filepath
        if (outfile is None) or (outfile == infile):
            # Backup original file
            infile = self.filepath + ".orig"
            os.rename(self.filepath, infile)
            outfile = self.filepath
        if os.path.exists(outfile):
            if overwrite:
                os.remove(outfile)
                logger.warning("Overwriting file: {0}".format(outfile))
            else:
                raise OSError("File already exists: {0}".format(outfile))
        subprocess.check_output(args=[
            "pdftk", infile, "update_info_utf8", "-", "output", outfile,
        ], input=self.text_metadata, universal_newlines=True)
        logger.info("Saved PDF file: {0}".format(outfile))


class BibEntry:
    """
    BibTeX database entries.
    """
    def __init__(self, filepath, ads):
        self.filepath = filepath
        self.ads = ads  # ADS querier
        self.pdf = PDF(self.filepath)
        self.doi = self.pdf.doi
        self.bibcode = self.pdf.bibcode
        self.arxiv = self.pdf.arxiv

    def query_ads(self):
        """
        Query the ADS to retrieve the bibliographic data
        """
        if self.pdf.bibcode:
            q = "bibcode:{0}".format(self.pdf.bibcode)
        elif self.pdf.doi:
            q = "doi:{0}".format(self.pdf.doi)
        elif self.pdf.arxiv:
            q = "arxiv:{0}".format(self.pdf.arxiv)
        else:
            raise ValueError("No available metadata in PDF for ADS query")
        logger.info("ADS query expression: q={0}".format(q))
        records = self.ads.query(q=q)
        if len(records) != 1:
            raise ValueError("Queried NO or MULTIPLE records")
        self.ads_record = records[0]
        logger.info("Queried record: {0}".format(str(self.ads_record)))

    def make_bib_data(self):
        """
        Make the BibTeX data for this entry from queried ADS record data
        """
        bibtype, persons, fields = self.ads_record.to_bib_data()
        fields["file"] = self.filepath
        self.bibtype = bibtype
        self.persons = persons
        self.fields = fields
        # Update DOI, BibCode and arXiv ID
        self.doi = fields.get("doi")
        self.bibcode = fields.get("adsbibcode")
        self.arxiv = fields.get("eprint")

    def make_bib_key(self, db=None):
        """
        Generate the BibTeX key for this entry from BibTeX data
        """
        first_author = self.persons["author"][0]
        last_name = "".join(first_author.last_names)
        last_name = unidecode(last_name)
        last_name = re.sub(r"[ {}`'\"\\]", "", last_name)
        year = self.fields["year"]
        journal = self.ads_record.get_bibstem()
        bibkey = "".join([last_name, year, journal])
        if db and db.exists_key(bibkey):
            num = 2
            while db.exists_key(bibkey+str(num)):
                num += 1
            bibkey += str(num)
        logger.info("Generated BibTeX key: {0}".format(bibkey))
        self.bibkey = bibkey

    def load_db_data(self, dbentry):
        self.bibkey = dbentry.key
        self.bibtype = dbentry.type
        self.persons = dbentry.persons
        self.fields = dbentry.fields
        # Update DOI, BibCode and arXiv ID
        self.doi = self.fields.get("doi")
        self.bibcode = self.fields.get("adsbibcode")
        self.arxiv = self.fields.get("eprint")

    def update_pdf(self):
        if self.doi:
            self.pdf.add_metadata("DOI", self.doi)
        if self.bibcode:
            self.pdf.add_metadata("AdsBibCode", self.bibcode)
        if self.arxiv:
            self.pdf.add_metadata("arXiv", self.arxiv)
        self.pdf.write()


class BibDB:
    """
    BibTeX database
    """
    modified = False

    def __init__(self, filepath):
        self.filepath = filepath
        self.db = pybtex.database.parse_file(filepath)
        logger.info("Loaded BibTeX database from: {0}".format(filepath))
        self.doi_bibkeys = self._make_bibkey_invmap("doi")
        self.bibcode_bibkeys = self._make_bibkey_invmap("adsbibcode")
        self.arxiv_bibkeys = self._make_bibkey_invmap("eprint")

    def _make_bibkey_invmap(self, key):
        """
        Make an inverse map of bibkey to the specified key,
        e.g., doi, adsbibcode, epring.
        """
        invmap = {
            entry.fields[key]: entry.key
            for entry in self.db.entries if entry.fields.get(key)
        }
        return invmap

    def exists_key(self, bibkey):
        return (bibkey in self.db.entries)

    def check_file(self):
        """
        Check the existence of entry-related file.
        """
        raise NotImplementedError

    def contains(self, entry):
        """
        Whether the given entry is contained in the database?
        Return the key of the entry if exists.
        """
        for key, invmap in [(entry.bibcode, self.bibcode_bibkeys),
                            (entry.doi, self.doi_bibkeys),
                            (entry.arxiv, self.arxiv_bibkeys)]:
            bibkey = invmap.get(key)
            if bibkey:
                return bibkey
        return None

    def get_entry(self, bibkey):
        return self.db.entries[bibkey]

    def update_entry(self, bibkey, entry):
        """
        Update the specified entry (e.g., the file path to PDF) if necessary.
        """
        dbentry = self.get_entry(bibkey)
        file_orig = dbentry.fields.get("file", "")
        if not PDF.is_annotated_of(file_orig, entry.filepath):
            dbentry.fields["file"] = entry.filepath
            self.modified = True
            logger.info("BibTeX DB: updated entry: <{0}>".format(bibkey))
        else:
            logger.info("Skip to update BibTeX entry: <{0}>".format(bibkey))

    def add_entry(self, entry):
        """
        Add new entry to the database
        """
        bibkey = entry.bibkey
        self.db.add_entry(bibkey,
                          Entry(type_=entry.bibtype, fields=entry.fields,
                                persons=entry.persons))
        self.modified = True
        if entry.bibcode:
            self.bibcode_bibkeys[entry.bibcode] = bibkey
        if entry.doi:
            self.doi_bibkeys[entry.doi] = bibkey
        if entry.arxiv:
            self.arxiv_bibkeys[entry.arxiv] = bibkey
        logger.info("BibTeX DB: added new entry: <{0}>".format(bibkey))

    def write(self, outfile=None, overwrite=False):
        """
        Write the database if modified.
        """
        if not self.modified:
            logger.info("Database not modified; skip")
            return
        #
        if outfile is None:
            outfile = self.filepath
            # Backup original file
            backfile = outfile + ".orig"
            os.rename(outfile, backfile)
        if os.path.exists(outfile):
            if overwrite:
                os.remove(outfile)
                logger.warning("Overwriting file: {0}".format(outfile))
            else:
                raise OSError("File already exists: {0}".format(outfile))
        self.db.to_file(outfile)
        logger.info("Saved database to file: {0}".format(outfile))


class FileIgnore:
    """
    Manage the files to be ignored.
    """
    def __init__(self, specfile):
        self.specfile = specfile
        spec = [line.strip() for line in open(specfile).readlines()]
        spec = [l for l in spec if (not l.startswith("#")) and (l != "")]
        # Remove prefix "./"
        self.spec = [s[2:] if s.startswith("./") else s for s in spec]
        logger.debug("Ignored specifications:\n{0}".format(
            "\n".join(self.spec)))

    def is_ignored(self, filepath):
        for s in self.spec:
            if fnmatch.fnmatch(filepath, s):
                logger.debug("Ignored file: {0}".format(filepath))
                return True
        return False


class FileCollection:
    """
    Build the collection of files to be processed.
    """
    def __init__(self, inputs, ignorespec=None):
        self.inputs = inputs
        paths = self._normalize(self._build(inputs))
        paths = self._filter(paths, ignorespec)
        self.paths = sorted(paths)
        logger.debug("Collection of {0} files:\n{1}".format(
            len(self.paths), "\n".join(self.paths)))

    @staticmethod
    def _build(inputs):
        paths = []
        for f in inputs:
            if os.path.isdir(f):
                # Walk through the directory and get all PDF files
                # Credit: https://stackoverflow.com/a/2186565/4856091
                for root, dirnames, filenames in os.walk(f):
                    for filename in fnmatch.filter(filenames, "*.pdf"):
                        paths.append(os.path.join(root, filename))
            elif f.endswith(".pdf"):
                paths.append(f)
            else:
                # Get the contents as list of files
                _files = [line.strip() for line in open(f).readlines()]
                _files = [_f for _f in _files
                          if (not _f.startswith("#")) and (_f != "")]
                paths += _files
        return paths

    @staticmethod
    def _normalize(paths):
        """
        Remove the prefix "./" from file paths
        """
        paths = [p[2:] if p.startswith("./") else p for p in paths]
        return paths

    @staticmethod
    def _filter(paths, ignorespec=None):
        if ignorespec is None:
            return paths
        ignore = FileIgnore(ignorespec)
        paths = [p for p in paths if not ignore.is_ignored(p)]
        return paths


def cmd_query(args):
    """
    Sub-command "query".
    """
    ads = AdsQuery(args.token)

    print("ADS query: q={0}".format(args.query))
    records = ads.query(q=args.query)
    if len(records) == 0:
        print("No records found on ADS!")
    else:
        for i, rec in enumerate(records):
            print("\n[%d/%d] %s" % (i+1, len(records), str(rec)))
            rec.pprint()


def cmd_pdf(args):
    """
    Sub-command "pdf".
    """
    pdf_missing_arxiv = set()
    pdf_missing_bibcode = set()
    pdf_missing_doi = set()
    # Find the PDF files that missing the specified metadata
    filepaths = FileCollection(args.inputs, args.ignore).paths
    # Update PDF metadata
    if (args.set_arxiv or args.set_bibcode or args.set_doi):
        if len(filepaths) != 1:
            raise ValueError("only allow ONE input PDF; " +
                             "but gave %d" % len(filepaths))
        pdf = PDF(filepaths[0])
        if args.set_arxiv:
            pdf.add_metadata("arXiv", args.set_arxiv)
        if args.set_bibcode:
            pdf.add_metadata("AdsBibCode", args.set_bibcode)
        if args.set_doi:
            pdf.add_metadata("DOI", args.set_doi)
        pdf.write()
    # Default action: show PDF information
    for fp in filepaths:
        print("\n==> PDF: {0}".format(fp))
        pdf = PDF(fp)
        print("Annotated: {0}".format(pdf.annotated))
        if pdf.doi:
            print("DOI: {0}".format(pdf.doi))
        else:
            logger.warning("DOI: missing!")
            pdf_missing_doi.add(fp)
        if pdf.bibcode:
            print("ADS BibCode: {0}".format(pdf.bibcode))
        else:
            logger.warning("ADS BibCode: missing!")
            pdf_missing_bibcode.add(fp)
        if pdf.arxiv:
            print("arXiv ID: {0}".format(pdf.arxiv))
        else:
            logger.info("arXiv ID: missing!")
            pdf_missing_arxiv.add(fp)
    # Show files missing the specified metadata
    if (args.find_missing_arxiv or
            args.find_missing_bibcode or
            args.find_missing_doi):
        label = [
            l for f, l in [(args.find_missing_arxiv,   "arXiv"),
                           (args.find_missing_bibcode, "BibCode"),
                           (args.find_missing_doi,     "DOI")]
            if f is True
        ]
        setlist = [
            s for f, s in [(args.find_missing_arxiv,   pdf_missing_arxiv),
                           (args.find_missing_bibcode, pdf_missing_bibcode),
                           (args.find_missing_doi,     pdf_missing_doi)]
            if f is True
        ]
        pdf_missing_metadata = set.intersection(*setlist)
        print("\n-----------------------------------------------------------")
        print("{0} PDF files missing metadata: {1}".format(
            len(pdf_missing_metadata), " && ".join(label)))
        print("\n".join(pdf_missing_metadata))


def cmd_db(args):
    """
    Sub-command "db".
    """
    db = BibDB(args.bibtex)
    ads = AdsQuery(args.token)
    #
    for filepath in args.input:
        logger.info("\n==> PDF: {0}".format(filepath))
        entry = BibEntry(filepath, ads=ads)
        bibkey = db.contains(entry)
        if bibkey:
            logger.info("Already in database: <{0}>".format(bibkey))
            db.update_entry(bibkey, entry)
            entry.load_db_data(db.get_entry(bibkey))
            entry.update_pdf()
        else:
            entry.query_ads()
            entry.make_bib_data()
            entry.make_bib_key(db)
            entry.update_pdf()
            db.add_entry(entry)
    #
    db.write()


def main():
    # Default arguments
    token_dft = ".adstoken"
    bibtex_dft = "references.bib"
    ignore_dft = ".adsignore"

    parser = argparse.ArgumentParser(
        description=("Maintain the BibTeX database of a collection of PDF " +
                     "papers by querying the ADS database"))
    # Common arguments
    parser.add_argument("-b", "--bibtex", dest="bibtex", default=bibtex_dft,
                        help="BibTeX database (default: %s)" % bibtex_dft)
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="report verbose debug information")
    parser.add_argument("-t", "--token", dest="token", default=token_dft,
                        help="file containing the ADS API token " +
                        "(default: %s)" % token_dft)
    parser.add_argument("-I", "--ignore", dest="ignore", default=ignore_dft,
                        help="ignore specification file " +
                        "(default: %s)" % ignore_dft)
    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")
    # Sub-command: "query"
    parser_query = subparsers.add_parser("query", help="Query the ADS service")
    parser_query.add_argument("query", help="query expression")
    parser_query.set_defaults(func=cmd_query)
    # Sub-command: "pdf"
    parser_pdf = subparsers.add_parser("pdf", help="Manipulate PDF files")
    parser_pdf.add_argument("--find-missing-arxiv",
                            dest="find_missing_arxiv",
                            action="store_true",
                            help="find PDF files missing arXiv ID")
    parser_pdf.add_argument("--find-missing-bibcode",
                            dest="find_missing_bibcode",
                            action="store_true",
                            help="find PDF files missing ADS BibCode")
    parser_pdf.add_argument("--find-missing-doi",
                            dest="find_missing_doi",
                            action="store_true",
                            help="find PDF files missing DOI")
    parser_pdf.add_argument("-a", "--set-arxiv", dest="set_arxiv",
                            help="set arXiv ID in PDF metadata")
    parser_pdf.add_argument("-b", "--set-bibcode", dest="set_bibcode",
                            help="set ADS BibCode in PDF metadata")
    parser_pdf.add_argument("-d", "--set-doi", dest="set_doi",
                            help="set DOI in PDF metadata")
    parser_pdf.add_argument("inputs", nargs="+",
                            help=("PDF files, directory, or text file of " +
                                  "filenames. NOTE: only one PDF file " +
                                  "when to update its metadata."))
    parser_pdf.set_defaults(func=cmd_pdf)
    # Sub-command: "db"
    parser_db = subparsers.add_parser("db", help="Maintain the BibTeX DB")
    parser_db.add_argument("-s", "--show", dest="show", action="store_true",
                           help="show database status")
    parser_db.add_argument("-c", "--check-pdf", dest="check_pdf",
                           action="store_true",
                           help="check the database against local PDF files")
    parser_db.add_argument("-a", "--add", dest="add", nargs="+",
                           help="add PDF files to the database")
    parser_db.set_defaults(func=cmd_db)
    #
    args = parser.parse_args()
    if (args.ignore == ignore_dft) and (not os.path.exists(ignore_dft)):
        args.ignore = None

    # Setup logging
    loglevel = "DEBUG" if args.debug else "INFO"
    logging.basicConfig(level=loglevel,
                        format="[%(levelname)s] <%(name)s> %(message)s")

    logger.debug("Parsed arguments:\n{0}".format(args))

    # Dispatch sub-commands to call its associated default function
    args.func(args)


if __name__ == "__main__":
    main()
