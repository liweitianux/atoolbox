#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Get the arxiv abstract data and PDF for a given arxiv id.
#
# Weitian LI <liweitianux@gmail.com>
# 2015/01/23
#

import sys
import re
import urllib
import subprocess
import time
import mimetypes

from bs4 import BeautifulSoup

mirror = "http://jp.arxiv.org/"

def get_url(arxiv_id):
    """
    Determine the full arxiv URL from the given ID/URL.
    """
    if re.match(r'^[0-9]{7}$', arxiv_id):
        print("ERROR: 7-digit ID not supported, please use the full URL")
        sys.exit(2)
    elif re.match(r'^[0-9]{4}\.[0-9]{4,5}$', arxiv_id):
        arxiv_url = mirror + "abs/" + arxiv_id
    elif re.match(r'^https{0,1}://.*arxiv.*/([0-9]{7}|[0-9]{4}\.[0-9]{4,5})$',
            arxiv_id):
        arxiv_url = arxiv_id
    elif re.match(r'[a-zA-Z0-9.-]*arxiv.*/([0-9]{7}|[0-9]{4}\.[0-9]{4,5})$',
            arxiv_id):
        arxiv_url = "http://" + arxiv_id
    else:
        print("ERROR: unknown arxiv ID: %s" % arxiv_id)
        exit(3)

    return arxiv_url


def get_id(arxiv_url):
    """
    Extract the ID from the URL.
    """
    return arxiv_url.split('/')[-1]


def get_arxiv_abstract(arxiv_url):
    """
    Get the arxiv abstract data and save to file '${id}.txt'.
    """
    request = urllib.request.urlopen(arxiv_url)
    arxiv_html = request.read()
    soup = BeautifulSoup(arxiv_html)
    title = soup.body.find('h1', attrs={'class': 'title'}).text\
            .replace('\n', ' ')
    authors = soup.body.find('div', attrs={'class': 'authors'}).text\
            .replace('\n', ' ')
    date = soup.body.find('div', attrs={'class': 'dateline'}).text\
            .strip('()')
    abstract = soup.body.find('blockquote', attrs={'class': 'abstract'})\
            .text.replace('\n', ' ')[1:]
    comments = soup.body.find('td', attrs={'class': 'comments'}).text
    subjects = soup.body.find('td', attrs={'class': 'subjects'}).text

    arxiv_id = get_id(arxiv_url)
    filename = arxiv_id + '.txt'
    f = open(filename, 'w')
    f.write("URL: %s\n" % arxiv_url)
    f.write("arXiv: %s\n" % arxiv_id)
    f.write("%s\n\n" % date)
    f.write("%s\n%s\n\n" % (title, authors))
    f.write("%s\n\n" % abstract)
    f.write("Comments: %s\n" % comments)
    f.write("Subjects: %s\n" % subjects)
    f.close()


def get_arxiv_pdf(arxiv_url):
    """
    Get the arxiv PDF with cURL.
    If the PDF is not generated yet, then retry after 10 seconds.
    """
    p = re.compile(r'/abs/')
    arxiv_pdf_url = p.sub('/pdf/', arxiv_url)
    arxiv_id = get_id(arxiv_url)
    filename = arxiv_id + '.pdf'
    cmd = 'curl -o %(filename)s %(url)s' %\
            {'filename': filename, 'url': arxiv_pdf_url}
    print("CMD: %(cmd)s" % {'cmd': cmd})
    subprocess.call(cmd, shell=True)
    output = subprocess.check_output(['file', '-ib', filename])
    filetype = output.decode(encoding='UTF-8').split(';')[0]
    pdftype = 'application/pdf'
    while filetype != pdftype:
        time.sleep(10)
        subprocess.call(cmd, shell=True)
        output = subprocess.check_output(['file', '-ib', filename])
        filetype = output.decode(encoding='UTF-8').split(';')[0]

def main():
    if len(sys.argv) != 2:
        print("Usage: %s <arxiv_id | arxiv_url>\n")
        sys.exit(1)

    arxiv_url = get_url(sys.argv[1])
    arxiv_id = get_id(arxiv_url)
    print("arxiv_url: %s" % arxiv_url)
    print("arxiv_id: %s" % arxiv_id)
    get_arxiv_abstract(arxiv_url)
    print("downloading pdf ...")
    get_arxiv_pdf(arxiv_url)


if __name__ == '__main__':
    main()

