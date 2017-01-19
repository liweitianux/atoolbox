#!/bin/bash
# PS4="##### "
# convert APE with cue file to FLAC tracks

selection=`zenity --title="请选择一个.ape文件和一个.cue文件" --file-selection --multiple`
if [ -z "$selection" ]; then exit 0; fi
if [[ "$selection" != *.ape\|*.cue && "$selection" != *.cue\|*.ape ]]; then exit 0; fi
if [ `echo $selection | gawk -F "|" 'END {print NF}'` -ge 3 ]; then exit 0; fi

arg1=${selection%|*}; arg2=${selection#*|}
if [[ "$arg1" == *.ape ]];then file_ape=$arg1; file_cue=$arg2
else file_ape=$arg2; file_cue=$arg1; fi

i=0
while [ -f "temp$i" ]; do
i=$(($i+1))
done
> temp$i
iconv -f gb18030 -t utf8 "$file_cue" > "temp$i" && mv "temp$i" "$file_cue"
rm -f "temp$i"

tracks=`gawk -vRS="TRACK" 'END {print NR-1}' "$file_cue"`
#echo $tracks

all_titles=`gawk -vRS='TRACK' -vFS='\n' \
'{j=0;for(i=1;i<=NF;i++){if($i~/TITLE/){print $i;j=1}}};j==0 {print "TITLE \"#####\""}' "$file_cue" | \
gawk -F "\"" 'NR>=2 {printf("%s|",$2)}'`

j=1; while [ -n "$all_titles" ]; do
title=${all_titles%%|*}
artist=${all_artists%%|*}
num=`printf "%0${#tracks}d" $j`
rm -f "${num}.$title.flac"
((j++))
all_titles=${all_titles#*|}
all_artists=${all_artists#*|}
done

shntool split -f "$file_cue" -t '%n.%t' -o flac "$file_ape" 2>&1 | \
gawk -vvar=$tracks 'NR>=1 {print "#正在转换...";print (NR-1)*100/var};{fflush();}' | \
zenity --title="进度" --text="正在转换..." --progress --auto-kill --auto-close

all_titles=`gawk -vRS='TRACK' -vFS='\n' \
'{j=0;for(i=1;i<=NF;i++){if($i~/TITLE/){print $i;j=1}}};j==0 {print "TITLE \"#####\""}' "$file_cue" | \
gawk -F "\"" 'NR>=2 {printf("%s|",$2)}'`

all_artists=`gawk -vRS='TRACK' -vFS='\n' \
'{j=0;for(i=1;i<=NF;i++){if($i~/PERFORMER/){print $i;j=1}}};\
j==0 {print "PERFORMER \"#####\""}' "$file_cue" | gawk -F "\"" 'NR>=2 {printf("%s|",$2)}'`

album=`gawk -vRS='TRACK' -vFS='\n' \
'{j=0;for(i=1;i<=NF;i++){if($i~/TITLE/){print $i;j=1}}};j==0 {print "TITLE \"#####\""}' "$file_cue" | \
gawk -F "\"" 'NR==1 {printf("%s",$2)}'`

j=1; while [ -n "$all_titles" ]; do
title=${all_titles%%|*}
artist=${all_artists%%|*}
num=`printf "%0${#tracks}d" $j`
if [ "$title" != "#####" ]; then metaflac --set-tag=TITLE="$title" "${num}.$title.flac"; fi
if [ "$artist" != "#####" ]; then metaflac --set-tag=ARTIST="$artist" "${num}.$title.flac"; fi
if [ "$album" != "#####" ]; then metaflac --set-tag=ALBUM="$album" "${num}.$title.flac"; fi
metaflac --set-tag=TRACKNUMBER="$j" "${num}.$title.flac"
#if [ "$title" != "#####" ]; then mv "${num}.$title.flac" "${num:1}.$title.flac"
#else mv "${num}.$title.flac" "${num:1}.NoTitle.flac"; fi
((j++))
all_titles=${all_titles#*|}
all_artists=${all_artists#*|}
done

zenity --info --text="结束。"
exit 0
