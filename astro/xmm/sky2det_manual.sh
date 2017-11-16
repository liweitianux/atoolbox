#!/bin/sh
#
# Convert from SKY(X,Y) coordinate to (DETX,DETY) coordinate for XMM.
# The required conversion coefficients are extracted from input FITS header.
#
#
# Weitian LI
# Created: 2015-11-09
# Updated: 2015-11-09
#

case "$1" in
    -[hH]*|--[hH]*)
        printf "usage:\n"
        printf "    `basename $0` evt=<evt> x=<sky_x> y=<sky_y>\n"
        exit 1
        ;;
esac

## functions {{{
# process commandline arguments
# cmdline arg format: `KEY=VALUE'
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

sky2wcs() {
    sky_val=$1
    crpx=$2
    crvl=$3
    cdlt=$4
    wcs_val=`python -c "import math; print(${crvl} + math.tan(${cdlt} * (${sky_val} - (${crpx}))))"`
    echo ${wcs_val}
}
## functions }}}

# process cmdline args using `getopt_keyval'
getopt_keyval "$@"

[ ! -e "${evt}" ] && echo "ERROR: ${evt} not exist" && exit 11

# Get the WCS conversion coefficients
fkeypar "${evt}" REFXCTYP
REFXCTYP=`pget fkeypar value`
fkeypar "${evt}" REFXCRPX
REFXCRPX=`pget fkeypar value`
fkeypar "${evt}" REFXCRVL
REFXCRVL=`pget fkeypar value`
fkeypar "${evt}" REFXCDLT
REFXCDLT=`pget fkeypar value`
fkeypar "${evt}" REFYCTYP
REFYCTYP=`pget fkeypar value`
fkeypar "${evt}" REFYCRPX
REFYCRPX=`pget fkeypar value`
fkeypar "${evt}" REFYCRVL
REFYCRVL=`pget fkeypar value`
fkeypar "${evt}" REFYCDLT
REFYCDLT=`pget fkeypar value`
echo "(X,Y) => (RA,DEC)"
echo "    RA  = ${REFXCRVL} + TAN[ (${REFXCDLT}) * (X - (${REFXCRPX})) ]"
echo "    DEC = ${REFYCRVL} + TAN[ (${REFYCDLT}) * (Y - (${REFYCRPX})) ]"

RA=`sky2wcs ${x} ${REFXCRPX} ${REFXCRVL} ${REFXCDLT}`
DEC=`sky2wcs ${y} ${REFYCRPX} ${REFYCRVL} ${REFYCDLT}`
echo "sky(${x},${y}) => wcs(${RA},${DEC})"

DETXY=`esky2det datastyle=user ra=${RA} dec=${DEC} outunit=det calinfostyle=set calinfoset="${evt}" | \
    grep -A 1 -E '^#\s+detX\s+detY' | tail -n 1`
detx=`echo ${DETXY} | awk '{ print $1 }'`
dety=`echo ${DETXY} | awk '{ print $2 }'`
echo "(RA,DEC) => DETXY(${detx},${dety})"

