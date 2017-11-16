#!/bin/sh
#
# Generate object and background spectra, and generate also RMF and ARF.
#
#
# Weitian LI
# Created: 2015-11-09
# Updated: 2015-11-09
#


case "$1" in
    -[hH]*|--[hH]*)
        printf "usage:\n"
        printf "    `basename $0` det=<detector> regtxt=<regtxt> rootname=<rootname> grpcnts=<group_mincounts>\n"
        exit 1
        ;;
esac

## functions {{{
# process commandline arguments
# cmdline arg format: `KEY=VALUE'
getopt_keyval() {
    until [ -z "$1" ]; do
        key=${1%%=*}                    # extract key
        val=${1#*=}                     # extract value
        keyval="${key}=\"${val}\""
        echo "## getopt: eval '${keyval}'"
        eval ${keyval}
        shift                           # shift, process next one
    done
}
## functions }}}

getopt_keyval "$@"

# Target
DET="${det}"
REGTXT="${regtxt}"
ROOTNAME="${rootname}"
GRPCNTS="${grpcnts:-50}"
echo "GRPCNTS: ${GRPCNTS}"

DET_TYPE=`echo "${DET}" | tr -d '12'`
PREFIX=`\ls ${DET}*-clean.fits | sed 's/^\(mos\|pn\)\([12]*S[0-9]\{3\}\).*$/\2/'`
[ ! -e "${REGTXT}" ] && echo "ERROR: ${REGTXT} not exist!" && exit 11

# clean-up previously generated files
#rm -fv ${DET_TYPE}${PREFIX}-corn.fits
rm -fv ${DET_TYPE}${PREFIX}-obj-im.fits
rm -fv ${DET_TYPE}${PREFIX}-exp-im.fits ${DET_TYPE}${PREFIX}-obj-im-sp-det.fits
rm -fv ${DET_TYPE}${PREFIX}-mask-im.fits
rm -fv ${DET_TYPE}${PREFIX}-obj.pi ${DET_TYPE}${PREFIX}-back.pi
rm -fv ${DET_TYPE}${PREFIX}.rmf ${DET_TYPE}${PREFIX}.arf

if [ "${DET}" = "pn" ]; then
    rm -fv ${DET_TYPE}${PREFIX}-obj-oot.pi ${DET_TYPE}${PREFIX}-obj-im-oot.fits
    #rm -fv ${DET_TYPE}${PREFIX}-corn-oot.pi
    SPEC_CMD="pn-spectra prefix=${PREFIX} region=${REGTXT} caldb=${SAS_ESAS_CALDB} mask=1 elow=0 ehigh=0 quad1=1 quad2=1 quad3=1 quad4=1"
    BACK_CMD="pn_back prefix=${PREFIX} caldb=${SAS_ESAS_CALDB} elow=0 ehigh=0 quad1=1 quad2=1 quad3=1 quad4=1"
else
    SPEC_CMD="mos-spectra prefix=${PREFIX} region=${REGTXT} caldb=${SAS_ESAS_CALDB} mask=1 elow=0 ehigh=0 ccd1=1 ccd2=1 ccd3=1 ccd4=1 ccd5=1 ccd6=1 ccd7=1"
    BACK_CMD="mos_back prefix=${PREFIX} caldb=${SAS_ESAS_CALDB} elow=0 ehigh=0 ccd1=1 ccd2=1 ccd3=1 ccd4=1 ccd5=1 ccd6=1 ccd7=1"
fi

eval ${SPEC_CMD}
eval ${BACK_CMD}

# Rename products
mv -v ${DET_TYPE}${PREFIX}-obj.pi             ${DET}_${ROOTNAME}.pi
mv -v ${DET_TYPE}${PREFIX}-back.pi            ${DET}_${ROOTNAME}_back.pi
mv -v ${DET_TYPE}${PREFIX}.rmf                ${DET}_${ROOTNAME}.rmf
mv -v ${DET_TYPE}${PREFIX}.arf                ${DET}_${ROOTNAME}.arf
mv -v ${DET_TYPE}${PREFIX}-obj-im-sp-det.fits ${DET}_${ROOTNAME}-sp.fits
mv -v ${DET_TYPE}${PREFIX}-spec.qdp           ${DET}_${ROOTNAME}-spec.qdp
mv -v ${DET_TYPE}${PREFIX}-aug.qdp            ${DET}_${ROOTNAME}-aug.qdp
if [ "${DET}" = "pn" ]; then
    mv -v ${DET_TYPE}${PREFIX}-obj-oot.pi ${DET}_${ROOTNAME}-oot.pi
    mv -v ${DET_TYPE}${PREFIX}-obj-os.pi ${DET}_${ROOTNAME}-os.pi
    mv -v ${DET_TYPE}${PREFIX}-aug-spec.qdp ${DET}_${ROOTNAME}-aug-spec.qdp
    mv -v ${DET_TYPE}${PREFIX}-aug-rev-hard.qdp ${DET}_${ROOTNAME}-aug-rev-hard.qdp
    mv -v ${DET_TYPE}${PREFIX}-aug-rev-rate.qdp ${DET}_${ROOTNAME}-aug-rev-rate.qdp
fi

# Group spectrum
specgroup spectrumset="${DET}_${ROOTNAME}.pi" groupedset="${DET}_${ROOTNAME}_grp.pi" rmfset="${DET}_${ROOTNAME}.rmf" arfset="${DET}_${ROOTNAME}.arf" mincounts=${GRPCNTS}

# Clean up again
rm -fv ${DET_TYPE}${PREFIX}-?ff.pi  ${DET_TYPE}${PREFIX}-?fc.pi
rm -fv ${DET_TYPE}${PREFIX}-?obj.pi ${DET_TYPE}${PREFIX}-?oc.pi
if [ "${DET}" = "pn" ]; then
    rm -fv ${DET_TYPE}${PREFIX}-?ff-oot.pi  ${DET_TYPE}${PREFIX}-?fc-oot.pi
    rm -fv ${DET_TYPE}${PREFIX}-?obj-oot.pi ${DET_TYPE}${PREFIX}-?oc-oot.pi
fi

