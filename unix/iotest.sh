#!/bin/sh
#
# Simple test to the disk write speed using `dd`.
#
# Aaron LI
# 2017-11-06
#

IN=/dev/zero
OUT=/tmp/iotest.$$
BS=16k
COUNT=16000

RESULT=${HOME}/iotest.txt

CMD="dd if=${IN} of=${OUT} bs=${BS} count=${COUNT} conv=sync"
SPEED=$(${CMD} 2>&1 | grep 'bytes/sec' | awk -F'[()]' '{ print $2 }')
rm ${OUT}
DATE=$(date +'%Y-%m-%d %H:%M')
echo "${DATE}  ${BS}*${COUNT}  ${SPEED}" >> ${RESULT}
