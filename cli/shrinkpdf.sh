#!/bin/sh
#
# Shrink the size of PDF files by adjust its quality using `gs` (GhostScript).
#
# Aaron LI
# 2013/09/18
#

case "$1" in
    -[hH]*|--[hH]*)
        printf "usage:\n"
        printf "    `basename $0` in=<input.pdf> out=<output.pdf> quality=<screen|ebook|printer|prepress> imgdpi=<img_dpi>\n"
        exit 1
        ;;
esac

getopt_keyval() {
    until [ -z "$1" ]
    do
        key=${1%%=*}                    # extract key
        val=${1#*=}                     # extract value
        keyval="${key}=\"${val}\""
        echo "## getopt: eval '${keyval}'"
        eval ${keyval}
        shift                           # shift, process next one
    done
}
getopt_keyval "$@"

if [ -z "${in}" ] || [ -z "${out}" ]; then
    printf "Error: 'in' or 'out' not specified\n"
    exit 2
fi
quality=${quality:-ebook}
imgdpi=${imgdpi:-120}

printf "# in: ${in}
# out: ${out}
# quality: ${quality}
# imgdpi: ${imgdpi}\n"

gs  -dNOPAUSE -dBATCH -dSAFER \
    -sDEVICE=pdfwrite \
    -dCompatibilityLevel=1.4 \
    -dPDFSETTINGS="/${quality}" \
    -dEmbedAllFonts=true \
    -dSubsetFonts=true \
    -dColorImageDownsampleType=/Bicubic \
    -dColorImageResolution=${imgdpi} \
    -dGrayImageDownsampleType=/Bicubic \
    -dGrayImageResolution=${imgdpi} \
    -dMonoImageDownsampleType=/Bicubic \
    -dMonoImageResolution=${imgdpi} \
    -sOutputFile=${out} \
    ${in}

