#!/bin/sh
#
# Copyright (c) 2013,2019 Aaron LI
# MIT License
#
# Compress a PDF file by adjust its quality using GhostScript.
#
# Credits:
# * https://www.ghostscript.com/doc/current/Use.htm
# * https://www.ghostscript.com/doc/current/VectorDevices.htm
#

usage() {
    cat << _EOF_
Compress a PDF file by adjust its quality using GhostScript.

usage:
    ${0##*/} [-d dpi] [-l level] [-s settings] <infile> <outfile>

options:
    -d dpi : image dpi to be downsampled to (default: 150)
    -l level : PDF compatibility level (default: 1.5)
    -s settings : predefined settings (default: ebook)
        valid choices: screen, ebook, printer, prepress, default
_EOF_

    exit 1
}

main() {
    local dpi=150
    local level=1.5
    local settings=ebook
    local infile outfile

    while getopts :hd:l:s: opt; do
        case ${opt} in
        h)
            usage
            ;;
        d)
            dpi=${OPTARG}
            ;;
        l)
            level=${OPTARG}
            ;;
        s)
            settings=${OPTARG}
            ;;
        \?)
            echo "Invalid option -${OPTARG}"
            usage
            ;;
        :)
            echo "Option -${OPTARG} requires an argument"
            usage
            ;;
        esac
    done

    shift $((${OPTIND} - 1))
    if [ $# -ne 2 ]; then
        usage
    fi

    infile="$1"
    outfile="$2"

    gs  -dQUIET -dNOPAUSE -dBATCH -dSAFER \
        -sDEVICE=pdfwrite \
        -dCompatibilityLevel=${level} \
        -dPDFSETTINGS=/${settings} \
        -dPrinted=false \
        -dEmbedAllFonts=true \
        -dSubsetFonts=true \
        -dFastWebView=true \
        -dColorImageDownsampleType=/Bicubic \
        -dColorImageResolution=${dpi} \
        -dGrayImageDownsampleType=/Bicubic \
        -dGrayImageResolution=${dpi} \
        -sOutputFile="${outfile}" \
        "${infile}"
}

main "$@"
