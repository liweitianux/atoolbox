#!/bin/sh
#
# Convert from SKY(X,Y) coordinate to (DETX,DETY) coordinate for XMM.
# Using the SAS tool "ecoordconv"
#
#
# Weitian LI
# Created: 2015-11-09
# Updated: 2015-11-09
#

if [ $# -ne 3 ]; then
    echo "usage:"
    echo "    `basename $0` <image> <sky.reg> <det.reg>"
    exit 1
fi

IMG="$1"
SKY_REG="$2"
DET_REG="$3"
[ -e "${DET_REG}" ] && mv -fv ${DET_REG} ${DET_REG}_bak

sky2det() {
    _img=$1
    _x=$2
    _y=$3
    _detxy=`ecoordconv imageset=${_img} x=${_x} y=${_y} coordtype=POS -w 0 -V 0 | \
        grep -E '^\s*DETX:\s*DETY:' | awk '{ print $3, $4 }'`
    echo "${_detxy}"
}


cat "${SKY_REG}" | while read line; do
    if echo "${line}" | grep -iq '^circle'; then
        # Across a circle region
        echo "CIRCLE: ${line}"
        CIRCLE_X=`echo "${line}" | awk -F'[(,)]' '{ print $2 }'`
        CIRCLE_Y=`echo "${line}" | awk -F'[(,)]' '{ print $3 }'`
        CIRCLE_R=`echo "${line}" | awk -F'[(,)]' '{ print $4 }'`
        DETXY=`sky2det ${IMG} ${CIRCLE_X} ${CIRCLE_Y}`
        CIRCLE_DETX=`echo "${DETXY}" | awk '{ print $1 }'`
        CIRCLE_DETY=`echo "${DETXY}" | awk '{ print $2 }'`
        CIRCLE_REG="circle(${CIRCLE_DETX},${CIRCLE_DETY},${CIRCLE_R})"
        echo "${CIRCLE_REG}" >> ${DET_REG}
    elif echo "${line}" | grep -iq '^physical'; then
        echo "detector" >> ${DET_REG}
    else
        # Just move to output region file
        echo "${line}" >> ${DET_REG}
    fi
done


