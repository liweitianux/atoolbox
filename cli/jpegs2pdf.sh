#!/bin/sh
#
#############################################################################
#  
#  Shellscript to convert a set of JPEG files to a multipage PDF.
#
#  Requirements: (1) Ghostscript needs to be installed on the local system.
#                (2) ImageMagick needs to be installed on the local system.
#
#  Usage:  jpegs2pdf.sh output.pdf file1.jpeg [file2.jpeg [file2.jpeg [...]]]
#
#  Copyright (c) 2007, <pipitas@gmail.com>
#                Use, distribute and modify without any restrictions.
#
#  Versions:
#          v1.0.0, Jul 12 2007:  initial version
#          v1.0.1, Jan 07 2011:  set viewJPEG.ps path (self-compiled GS 9.02)
#
#############################################################################

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "    `basename $0` output.pdf 1.jpg ..."
    exit 1
fi

outfile=$1
shift

param=""
for i in "$@" ; do
   dimension=$(identify -format "%[fx:(w)] %[fx:(h)]" "${i}")
   param="${param} <</PageSize [${dimension}]>> setpagedevice (${i}) viewJPEG showpage"
done

gs \
  -sDEVICE=pdfwrite \
  -dPDFSETTINGS=/prepress \
  -o "$outfile" \
  viewjpeg.ps \
  -c "${param}"

