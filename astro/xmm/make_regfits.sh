#/bin/sh
#
# Make a new region FITS file based on the original bkg-region, which
# is created by SAS tool 'region', and custom ds9 region list.
# The newly created region FITS file can be used to create a new mask.
#
# Aaron LI
# Created: 2015-11-06
# Updated: 2015-11-09
#
# ChangeLog:
#   * Update sed match pattern to allow '-' minus sign
#

if [ $# -ne 3 ]; then
    echo "Usage:"
    echo "    `basename $0` <orig_reg.fits> <ds9.reg> <new_reg.fits>"
    echo ""
    echo "Note: only CIRCLE region are supported!"
    exit 1
fi

ORIG_REGFITS="$1"
DS9REG="$2"
NEW_REGFITS="$3"
[ ! -e "${ORIG_REGFITS}" ] && \
    echo "ERROR: ${ORIG_REGFITS} does not exist" && exit 11
[ ! -e "${DS9REG}" ] && echo "ERROR: ${DS9REG} does not exist" && exit 12
[ -e "${NEW_REGFITS}" ] && mv -fv ${NEW_REGFITS} ${NEW_REGFITS}_bak


TMP_CDFILE="cdfile.tmp$$"
TMP_DATAFILE="datafile.tmp$$"

flcol "${ORIG_REGFITS}[REGION]" > ${TMP_CDFILE}
sed -i'' '/^___Column_Names__/d' ${TMP_CDFILE}

grep -i '^circle' ${DS9REG} | tr 'a-z' 'A-Z' | \
    sed -e 's/\(CIRCLE\)(\([0-9.-]\+\),\([0-9.-]\+\),\([0-9.-]\+\))/"!\1"  \2 0 0 0  \3 0 0 0  \4 0 0 0  0 0 0 0  1/' | \
    tr '"' "'" > ${TMP_DATAFILE}

fcreate cdfile=${TMP_CDFILE} datafile=${TMP_DATAFILE} \
    outfile=${NEW_REGFITS} tbltype="binary" extname="REGION"

cphead "${ORIG_REGFITS}[Primary]" "${NEW_REGFITS}[Primary]"
cphead "${ORIG_REGFITS}[REGION]" "${NEW_REGFITS}[REGION]"

rm -f ${TMP_CDFILE} ${TMP_DATAFILE}

