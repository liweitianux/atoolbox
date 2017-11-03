#!/bin/sh
#
# Convert FLV to MP4 with ffmpeg and preserving original encodings.
#
# Credit:
# [1] FFMPEG convert flv to mp4 without losing quality
#     http://superuser.com/a/727081
#
# Aaron LI
# Created: 2015-12-28
#

if [ $# -ne 1 ] && [ $# -ne 2 ]; then
    echo "Usage: `basename $0` <file.flv> [file.mp4]"
    exit 1
fi

INFILE="$1"
OUTFILE="$2"
[ -z "${OUTFILE}" ] && OUTFILE="${INFILE%.flv}.mp4"

ffmpeg -i "${INFILE}" -codec copy "${OUTFILE}"

