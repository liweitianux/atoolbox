#!/bin/sh
#
# Merge multiple PDFs with pdftk.
#
# Ref:
# Merging Multiple PDFs under GNU/Linux
# https://blog.dbrgn.ch/2013/8/14/merge-multiple-pdfs/
#
# Weitian LI
# 2015/01/23
#

if [ $# -lt 2 ]; then
    printf "Usage: `basename $0` out.pdf in1.pdf ...\n"
    exit 1
fi

outpdf="$1"
shift

echo "Input files: $@"
pdftk "$@" cat output "${outpdf}"

