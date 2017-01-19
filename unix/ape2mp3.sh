#!/bin/sh

###                                                ###
# For APE & CUE file --> Mp3                         #
# Version : 0.20                                     #
# This script build high quality mp3 from APE.       #
# You just run it in the dir include APE & CUE file. #
#                                                    #
# Author : kevin.li823@gmail.com                     #
###                                                ###

echo Lame is the best mp3 encoder!
export APE_FILE=`ls *.[aA][pP][eE]`
export CUE_FILE=`ls *.[cC][uU][eE]`
export WAV_FILE=${APE_FILE%ape}wav
export UTF_CUE_FILE='UTF.cue'

ALBUM=
CD_ACTOR=
GENRE=
DATE=2000
ACTOR=
TRACK=
TITLE=

# Tools have been Installed
which dos2unix >/dev/null 2>&1 || (echo No dos2unix in System; exit )
which mac >/dev/null 2>&1 || (echo No mac in System; exit )
which bchunk >/dev/null 2>&1 || (echo No bchunk in System; exit )
which iconv >/dev/null 2>&1 || (echo No iconv in System; exit )
which lame >/dev/null 2>&1 || (echo No lame in System; exit )

# DOS -> Unix
dos2unix $CUE_FILE

# GBK -> UTF8
iconv -f gbk -t utf8 $CUE_FILE > $UTF_CUE_FILE

# Monkey's Audio Codec, ape -> wav
echo Decoding APE file, it may take a few minutes.
echo Please wait...
mac $APE_FILE $WAV_FILE -d
echo "APE --> WAV Complete! "

# Split wav with cue file
echo "Start to split wav! "
bchunk $WAV_FILE $CUE_FILE -w wav_
echo "Splitting Done"

# Scan Album's Information
ALBUM=`grep 'TITLE' $CUE_FILE|sed -n '1p'|awk -F \" '{print $2}'|\
                                                      sed s/\ /_/g`
UTF_ALBUM=`grep 'TITLE' $UTF_CUE_FILE|sed -n '1p'|awk -F \" '{print $2}'|\
                                                      sed s/\ /_/g`
CD_ACTOR=`grep 'PERFORMER' $CUE_FILE|sed -n '1p'|awk -F \" '{print $2}'|\
                                                      sed s/\ /_/g`
GENRE=`grep ^REM $CUE_FILE|awk '{print ($2, $3)}'|grep ^GENRE |sed -n '1p'|\
                                             awk '{print $2}'|\sed s/\ /_/g`
DATE=`grep ^REM $CUE_FILE|awk '{print ($2, $3)}'|grep ^DATE |sed -n '1p'|\
                                           awk '{print $2}'|\sed s/\ /_/g`
# Add Album's Information to CMD line
[ -n $ALBUM ] && CD_INFO="--tl ${ALBUM} "
[ -n $GENRE ] && CD_INFO="${CD_INFO}""--tg ${GENRE} "
[ -n $DATE ] && CD_INFO="${CD_INFO}""--ty $i{DATE}"

for i in `ls wav_*`
do
    # Scan Songs' Information
    TRACK=`echo $i|sed s/wav_//g|sed s/.wav//g`
    TITLE=`sed -n "/TRACK[\ \   ]"$TRACK"/,$"p $CUE_FILE|grep TITLE|\
                     sed -n '1p'|awk -F \" '{print $2}'|sed s/\ /_/g`
    UTF_TITLE=`sed -n "/TRACK[\ \   ]"$TRACK"/,$"p $UTF_CUE_FILE|grep TITLE|\
                     sed -n '1p'|awk -F \" '{print $2}'|sed s/\ /_/g`
    ACTOR=`sed -n "/TRACK[\ \   ]"$TRACK"/,$"p $CUE_FILE|grep PER|\
                     sed -n '1p'|awk -F \" '{print $2}'|sed s/\ /_/g`

    # Add Actor to CMDline if possible
    [ -z $ACTOR ] && ACTOR=$CD_ACTOR

    # Add Songs' Information to CMD line
    [ -n $TITLE ] && NAME=`echo $TRACK\_$UTF_TITLE.mp3`
    [ -n $TRACK ] && LAME_INFO=" --tn $TRACK "
    [ -n $TITLE ] && LAME_INFO="$LAME_INFO --tt $TITLE "
    [ -n $ACTOR ] && LAME_INFO="$LAME_INFO --ta $ACTOR "
    LAME_INFO="${LAME_INFO} --tc Made_by_Kevin_Li"

    echo
    echo "========== Now Laming $UTF_TITLE =========="

    #lame -b 32 -m j -h -V 0 -B 320 -q 0 --noath -k --id3v2-only --tt\
    #      $TITLE --ta $ACTOR --tl $ALBUM --tc Made_by_Kevin_Li --tn\
    #      $TRACK $i $NAME
   
    # Now Lame the Songs
    lame -b 32 -m j -h -V 0 -B 320 -q 0 --noath -k --id3v2-only \
                                       $CD_INFO $LAME_IMFO $i $NAME
   
    LAME_INFO=

    echo "========== Lame $UTF_TITLE Done!  =========="
    echo
done

# File Arrangement
[ -n `ls *.mp3` ] && rm *.wav
rm $UTF_CUE_FILE
mkdir \[$ACTOR_$UTF_ALBUM\_MP3\]
mv *.mp3 \[$ACTOR_$UTF_ALBUM\_MP3\]
echo
echo "===================== All Done! ======================="
[/code]
