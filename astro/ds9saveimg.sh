#!/bin/sh
#
# Copyright (c) 2019 Aaron LI <aly@aaronly.me>
# MIT License
#
# Use SAOImage ds9 to open a FITS image, load the regions from a region
# file, and then snapshot the window content to an image file via XPA.
#
# References:
# - http://ds9.si.edu/doc/ref/xpa.html#saveimage
#

NAME="${0##*/}"

if [ $# -ne 2 ]; then
    echo "usage: ${NAME} <file.fits> <ds9.reg>"
    exit 1
fi

FITS="$1"
REG="$2"
IMG="${FITS%.fits}.png"
TITLE="${NAME%.sh}$$"

ds9 ${FITS} \
    -width 768 -height 768 \
    -zoom to fit \
    -scale linear -scale mode 99.9 \
    -cmap sls \
    -regions format ds9 -regions ${REG} \
    -title ${TITLE} &
PID=$!
echo "Launched ds9 as PID=${PID}"

retry=1
while [ ${retry} -ne 0 ]; do
    sleep 1
    # Bring the window to the front and snapshot to an image file
    wmctrl -a ${TITLE}
    xpaset -p ${TITLE} saveimage png ${IMG}
    retry=$?
done
echo "${TITLE}: ${FITS} + ${REG} => ${IMG}"

#echo 'paused ...' && read _
kill ${PID}
