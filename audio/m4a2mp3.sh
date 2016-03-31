#!/bin/sh

bitrate=192

for i in *.m4a; do
    faad -o - "$i" | lame -h -b $bitrate - "${i%m4a}mp3"
done

