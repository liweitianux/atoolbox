#!/bin/sh
#
##################################################################
## filter, remove source, gti, etc.                             ##
## Ref: http://heasarc.gsfc.nasa.gov/docs/xmm/abc/node8.html    ##
##                                                              ##
## LIweitiaNux, February 17, 2012                               ##
##################################################################

usage() {
    printf "Usage:\n"
    printf "     `basename $0` broad|soft|hard [E_START:E_END]\n"
}

## process parameters
if [ $# -eq 1 ]; then
    E_BAND=$1
    if [ "${E_BAND}" = "broad" ]; then
        E_RANGE=300:10000
    elif [ "${E_BAND}" = "soft" ]; then
        E_RANGE=300:2000
    elif [ "${E_BAND}" = "hard" ]; then
        E_RANGE=2000:10000
    elif [ "${E_BAND}" = "xid" ]; then
        E_RANGE=500:4500
    else
        printf "ERROR: unknown energy name\n"
        usage
        exit 1
    fi
elif [ $# -eq 2 ]; then
    E_BAND=$1
    E_RANGE=$2
else
    printf "ERROR: no parameters given\n"
    usage
    exit 2
fi
# energy range
PIMIN=`echo ${E_RANGE} | cut -d':' -f1`
PIMAX=`echo ${E_RANGE} | cut -d':' -f2`
if [ "x${PIMIN}" = "x" ] || [ "x${PIMAX}" = "x" ]; then
    printf "ERROR: given parameters NOT match\n"
    usage
    exit 3
fi
# get 'ecf'
read -p "    ECF of MOS1 for this energy band: " ECF_M1
read -p "    ECF of MOS2 for this energy band: " ECF_M2
if [ "x${ECF_M1}" = "x" ] || [ "x${ECF_M2}" = "x" ]; then
    printf "ERROR: ECF given error\n"
    exit 10
fi

WD=`pwd -P`
VARS_FILE=${WD}/variables
if ! [ -e ${VARS_FILE} ]; then
    printf "ERROR: NOT found '${VARS_FILE}' in the current directory\n"
    printf "    You may in wrong directory\n"
    exit 1
fi
## load previous saved variables
. ${VARS_FILE}
printf "variables loaded from file '${VARS_FILE}'\n"

## link needed files
printf "link needed files ...\n"
cd ${PROC}
[ -e ${ATTHK} ] || ln -sv ../PPS/${ATTHK} .
[ -e ${MOS1_EVT} ] || ln -sv ../PPS/${MOS1_EVT} .
[ -e ${MOS2_EVT} ] || ln -sv ../PPS/${MOS2_EVT} .

## filter the original event file
printf "filter by energy on the original event file ...\n"
M1_EVT="${OBS_ID}_mos1.evt"
M2_EVT="${OBS_ID}_mos2.evt"
evselect table=${MOS1_EVT} withfilteredset=yes \
    filteredset=${M1_EVT} filtertype=expression \
    expression="(PATTERN<=12) && (PI in [200:12000]) && (FLAG==0) && #XMMEA_EM" \
    keepfilteroutput=yes updateexposure=yes filterexposure=yes
evselect table=${MOS2_EVT} withfilteredset=yes \
    filteredset=${M2_EVT} filtertype=expression \
    expression="(PATTERN<=12) && (PI in [200:12000]) && (FLAG==0) && #XMMEA_EM" \
    keepfilteroutput=yes updateexposure=yes filterexposure=yes

## make images
printf "bin event file to make image ...\n"
M1_IMG="${OBS_ID}_mos1_${E_BAND}.img"
M2_IMG="${OBS_ID}_mos2_${E_BAND}.img"
evselect table=${M1_EVT} withimageset=yes imageset=${M1_IMG} \
    imagebinning=binSize xcolumn=X ximagebinsize=22 \
    ycolumn=Y yimagebinsize=22 filtertype=expression \
    expression="(FLAG==0) && (PI in [${E_RANGE}])"
evselect table=${M2_EVT} withimageset=yes imageset=${M2_IMG} \
    imagebinning=binSize xcolumn=X ximagebinsize=22 \
    ycolumn=Y yimagebinsize=22 filtertype=expression \
    expression="(FLAG==0) && (PI in [${E_RANGE}])"

## make exposure map
printf "make exposure map ...\n"
M1_EXP="${OBS_ID}_mos1_${E_BAND}_exp.img"
M2_EXP="${OBS_ID}_mos2_${E_BAND}_exp.img"
eexpmap attitudeset=${ATTHK} eventset=${M1_EVT} imageset=${M1_IMG} \
    expimageset=${M1_EXP} pimin=${PIMIN} pimax=${PIMAX}
eexpmap attitudeset=${ATTHK} eventset=${M1_EVT} imageset=${M2_IMG} \
    expimageset=${M2_EXP} pimin=${PIMIN} pimax=${PIMAX}

## make detection mask for detector
printf "make detection mask for detector ...\n"
M1_MASK="${OBS_ID}_mos1_${E_BAND}_mask.img"
M2_MASK="${OBS_ID}_mos2_${E_BAND}_mask.img"
emask expimageset=${M1_EXP} detmaskset=${M1_MASK}
emask expimageset=${M2_EXP} detmaskset=${M2_MASK}

## 'local mode' sliding box detection
## may need to increase parameter 'imagebuffersize', say 'imagebuffersize=2000'
printf "sliding box detection, in *local* mode ...\n"
M1_BOXLIST_L="${OBS_ID}_mos1_${E_BAND}_boxlist_local.fits"
M2_BOXLIST_L="${OBS_ID}_mos2_${E_BAND}_boxlist_local.fits"
eboxdetect usemap=no likemin=8 withdetmask=yes detmasksets=${M1_MASK} \
    imagesets=${M1_IMG} expimagesets=${M1_EXP} pimin=${PIMIN} pimax=${PIMAX} \
    boxlistset=${M1_BOXLIST_L}
eboxdetect usemap=no likemin=8 withdetmask=yes detmasksets=${M2_MASK} \
    imagesets=${M2_IMG} expimagesets=${M2_EXP} pimin=${PIMIN} pimax=${PIMAX} \
    boxlistset=${M2_BOXLIST_L}

## generate spline background map from the non-source regions
printf "generate background map from the non-source regions ...\n"
M1_BKG="${OBS_ID}_mos1_${E_BAND}_bkg.img"
M2_BKG="${OBS_ID}_mos2_${E_BAND}_bkg.img"
esplinemap bkgimageset=${M1_BKG} imageset=${M1_IMG} scut=0.005 \
    boxlistset=${M1_BOXLIST_L} nsplinenodes=16 withdetmask=yes \
    detmaskset=${M1_MASK} withexpimage=yes expimageset=${M1_EXP}
esplinemap bkgimageset=${M2_BKG} imageset=${M2_IMG} scut=0.005 \
    boxlistset=${M2_BOXLIST_L} nsplinenodes=16 withdetmask=yes \
    detmaskset=${M2_MASK} withexpimage=yes expimageset=${M2_EXP}

## sliding box source detection again, in *map* mode
## improve the detection sensitivity
## also NOTE the parameter 'imagebuffersize'
printf "sliding box detection again, in *map* mode ...\n"
M1_BOXLIST_M="${OBS_ID}_mos1_${E_BAND}_boxlist_map.fits"
M2_BOXLIST_M="${OBS_ID}_mos2_${E_BAND}_boxlist_map.fits"
eboxdetect usemap=yes likemin=8 withdetmask=yes detmasksets=${M1_MASK} \
    imagesets=${M1_IMG} expimagesets=${M1_EXP} pimin=${PIMIN} \
    pimax=${PIMAX} boxlistset=${M1_BOXLIST_M} \
    bkgimagesets=${M1_BKG}
eboxdetect usemap=yes likemin=8 withdetmask=yes detmasksets=${M2_MASK} \
    imagesets=${M2_IMG} expimagesets=${M2_EXP} pimin=${PIMIN} \
    pimax=${PIMAX} boxlistset=${M2_BOXLIST_M} \
    bkgimagesets=${M2_BKG}

## final source list
## The energy conversion values (ECFs) can be supplied to convert the source
## count rates into fluxes. The ECFs for each detector and energy band depend
## on the pattern selection and filter used during the observation. For more
## information, please consult the calibration paper ``SSC-LUX-TN-0059'',
## available at the XMM-Newton Science Operations Center or see Table 3.2 in
## the '2XMM Catalogue User Guide'.
printf "final source list ...\n"
M1_EMLLIST="${OBS_ID}_mos1_${E_BAND}_emllist.fits"
M2_EMLLIST="${OBS_ID}_mos2_${E_BAND}_emllist.fits"
emldetect imagesets=${M1_IMG} expimagesets=${M1_EXP} bkgimagesets=${M1_BKG} \
    pimin=${PIMIN} pimax=${PIMAX} boxlistset=${M1_BOXLIST_M} \
    ecf=${ECF_M1} mlmin=10.0 mllistset=${M1_EMLLIST}
emldetect imagesets=${M2_IMG} expimagesets=${M2_EXP} bkgimagesets=${M2_BKG} \
    pimin=${PIMIN} pimax=${PIMAX} boxlistset=${M2_BOXLIST_M} \
    ecf=${ECF_M2} mlmin=10.0 mllistset=${M2_EMLLIST}

## optional, make a sensitivity map
printf "optional, make a sensitivity map ...\n"
M1_SENS="${OBS_ID}_mos1_${E_BAND}_sens.img"
M2_SENS="${OBS_ID}_mos2_${E_BAND}_sens.img"
esensmap expimagesets=${M1_EXP} bkgimagesets=${M1_BKG} \
    detmasksets=${M1_MASK} sensimageset=${M1_SENS} mlmin=10.0
esensmap expimagesets=${M2_EXP} bkgimagesets=${M2_BKG} \
    detmasksets=${M2_MASK} sensimageset=${M2_SENS} mlmin=10.0

## write source list to 'ds9 reg' file and view
M1_SRC="${OBS_ID}_mos1_${E_BAND}_src.reg"
M2_SRC="${OBS_ID}_mos2_${E_BAND}_src.reg"
srcdisplay boxlistset=${M1_EMLLIST} imageset=${M1_IMG} \
    withregionfile=yes regionfile=${M1_SRC} sourceradius=0.003
srcdisplay boxlistset=${M2_EMLLIST} imageset=${M2_IMG} \
    withregionfile=yes regionfile=${M2_SRC} sourceradius=0.003

## remove duplicated lines in 'regionfile'
printf "remove duplicated lines in reg files ...\n"
mv ${M1_SRC} ${M1_SRC}_tmp
mv ${M2_SRC} ${M2_SRC}_tmp
cat ${M1_SRC}_tmp | uniq > ${M1_SRC}
cat ${M2_SRC}_tmp | uniq > ${M2_SRC}
rm ${M1_SRC}_tmp ${M2_SRC}_tmp

## save variables
printf "save variables ...\n"
CHECK="`grep 'EXP' ${VARS_FILE}`"
if [ "x${CHECK}" = "x" ]; then
    printf "# `date`\n" >> ${VARS_FILE}
    printf "M1_EVT=${M1_EVT}\n" >> ${VARS_FILE}
    printf "M1_IMG=${M1_IMG}\n" >> ${VARS_FILE}
    printf "M1_BKG=${M1_BKG}\n" >> ${VARS_FILE}
    printf "M1_EXP=${M1_EXP}\n" >> ${VARS_FILE}
    printf "M1_SENS=${M1_SENS}\n" >> ${VARS_FILE}
    printf "M2_EVT=${M2_EVT}\n" >> ${VARS_FILE}
    printf "M2_IMG=${M2_IMG}\n" >> ${VARS_FILE}
    printf "M2_BKG=${M2_BKG}\n" >> ${VARS_FILE}
    printf "M2_EXP=${M2_EXP}\n" >> ${VARS_FILE}
    printf "M2_SENS=${M2_SENS}\n" >> ${VARS_FILE}
    printf "\n" >> ${VARS_FILE}
fi

