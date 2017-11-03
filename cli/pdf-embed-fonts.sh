#!/bin/sh
#
# Embed a PDF file with all fonts it uses, so it can be rendered and printed
# correctly by every PDF reader on all types of OS.
#
#
# Credits:
# [1] The PDF viewer 'Evince' on Linux can not display some math symbols correctly
#     http://stackoverflow.com/a/10282269/4856091
#
#
# Aaron LI
# 2015-10-24
#

usage() {
    echo "usage: `basename $0` <input.pdf> [ output_embedfonts.pdf  ]"
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi
case "$1" in
    -[hH]*)
        usage
        ;;
esac

INFILE="$1"
OUTFILE="$2"

[ -z "${OUTFILE}" ] && OUTFILE="${INFILE%.pdf}_embedfonts.pdf"

# "-dPDFSETTINGS=/prepress" tells Ghostscript to embed all non-embedded fonts.
gs -o "${OUTFILE}" -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite "${INFILE}"

