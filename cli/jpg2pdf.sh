#!/bin/sh
#
# Copyright (c) 2019 Aaron LI <aly@aaronly.me>
# MIT License
#
# Convert a set of JPEG files to a multipage PDF using Ghostscript and
# ImageMagick (requires the "identify" command).
#
# Credit: pipitas@gmail.com
#

if [ $# -lt 2 ]; then
    echo "usage: ${0##*/} output.pdf 1.jpg ..."
    exit 1
fi

outfile="$1"
shift

fcmd=$(mktemp)
N=$#
i=0
echo "Total images: ${N}"
while [ -n "$1" ]; do
    f="$1"; shift; i=$((${i} + 1))
    echo -n "${i}.."
    dim=$(identify -format "%[fx:(w)] %[fx:(h)]" "${f}")
    echo "<</PageSize [${dim}]>> setpagedevice (${f}) viewJPEG showpage" \
        >> ${fcmd}
done

gs \
  -sDEVICE=pdfwrite \
  -dPDFSETTINGS=/prepress \
  -o "${outfile}" \
  viewjpeg.ps \
  -f ${fcmd}

rm -f ${fcmd}
echo "Output file: ${outfile}"

