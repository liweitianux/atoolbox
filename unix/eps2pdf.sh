#!/bin/sh

# convert from eps to pdf using gs

GS="`which gs`"
BASENAME="`basename $1 .eps`"
${GS} -sDEVICE=pdfwrite -dNOPAUSE -dQUIET -sOutputFile=${BASENAME}.pdf - < $1

