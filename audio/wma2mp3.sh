#!/bin/sh
# convert *.wma to *.mp3

current_directory=$( pwd )

#remove spaces
for i in *.wma; do mv "$i" `echo $i | tr ' ' '_'`; done

#remove uppercase
for i in *.[Ww][Mm][Aa]; do mv "$i" `echo $i | tr '[A-Z]' '[a-z]'`; done

#Rip with Mplayer / encode with LAME
for i in *.wma ; do mplayer -vo null -vc dummy -af resample=44100 -ao pcm
    -waveheader $i && lame -m s audiodump.wav -o $i; done

#convert file names
for i in *.wma; do mv "$i" "`basename "$i" .wma`.mp3"; done

rm audiodump.wav

