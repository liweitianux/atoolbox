#!/bin/sh
#
##################################################################
## reprocess data                                               ##
## apply the most recent calibrations                           ##
##                                                              ##
## LIweitiaNux, February 17, 2012                               ##
##################################################################

## init SAS if needed
if ! echo $PATH | tr ':' '\n' | grep 'xmmsas' > /dev/null; then
    sasinit > /dev/null     # 'sasinit' is a func defined in ~/.bashrc
fi

## set needed variables
WD=`pwd -P`
ODF=${WD}/ODF
if ! [ -d ${ODF} ]; then
    printf "ERROR: NOT found 'ODF' directory in the current place\n"
    printf "    You may in wrong directory\n"
    exit 1
fi
export SAS_ODF=${ODF}
export SAS_CCF=${ODF}/ccf.cif
PPS=${WD}/PPS       # dir to save processed files
PROC=${WD}/PROC     # processing dir
[ -d ${PPS} ] && rm -f ${PPS}/* || mkdir ${PPS}
[ -d ${PROC} ] && rm -f ${PROC}/* || mkdir ${PROC}
printf "ODF=${ODF}\n"
printf "PPS=${PPS}\n"
printf "PROC=${PROC}\n"

## prepare the data
## Ref: http://heasarc.gsfc.nasa.gov/docs/xmm/abc/node7.html
cd ${ODF}
[ -e ./ccf.cif ] && rm -f ./ccf.cif
cifbuild        # generate new 'ccf.cif'
if ls *SUM.SAS; then
    rm -f *SUM.SAS
fi
odfingest       # generate new '*SUM.SAS'
export SAS_ODF=${ODF}/`ls *SUM.SAS`

## rerunning the pipeline
## Ref: http://heasarc.gsfc.nasa.gov/docs/xmm/abc/node8.html
cd ${PPS}
printf "enter directory ${PPS}\n"
printf "rerunning the pipeline ...\n"
printf "run 'emproc' to processe MOS* data ...\n"
emproc      # process 'MOS*' data
printf "run 'epproc' to processe PN data ...\n"
epproc      # process 'PN' data

OBS_ID=`ls *MOS1*ImagingEvts* | sed 's/_EMOS.*//' | sed 's/[0-9]*_//'`
MOS1_EVT=`ls *EMOS1*ImagingEvts*`
MOS2_EVT=`ls *EMOS2*ImagingEvts*`
PN_EVT=`ls *PN*ImagingEvts*`
ATTHK=`ls *AttHk*`

## store variables for later use
VARS_FILE=${WD}/variables
printf "save some variables to file '${VARS_FILE}' for later use ...\n"
[ -e ${VARS_FILE} ] && mv -f ${VARS_FILE} ${VARS_FILE}.bak
touch ${VARS_FILE}
printf "#!/bin/sh\n\n" >> ${VARS_FILE}
printf "# `date`\n\n" >> ${VARS_FILE}
printf "export SAS_CCFPATH=${SAS_CCFPATH}\n" >> ${VARS_FILE}
printf "export SAS_CCF=${SAS_CCF}\n" >> ${VARS_FILE}
printf "export SAS_ODF=${SAS_ODF}\n\n" >> ${VARS_FILE}
printf "WORK_DIR=${WD}\n" >> ${VARS_FILE}
printf "ODF=${ODF}\n" >> ${VARS_FILE}
printf "PPS=${PPS}\n" >> ${VARS_FILE}
printf "PROC=${PROC}\n\n" >> ${VARS_FILE}
printf "OBS_ID=${OBS_ID}\n" >> ${VARS_FILE}
printf "MOS1_EVT=${MOS1_EVT}\n" >> ${VARS_FILE}
printf "MOS2_EVT=${MOS2_EVT}\n" >> ${VARS_FILE}
printf "PN_EVT=${PN_EVT}\n" >> ${VARS_FILE}
printf "ATTHK=${ATTHK}\n\n" >> ${VARS_FILE}
printf "DONE\n"

printf "!!!ALL FINISHED!!!
exit 0

