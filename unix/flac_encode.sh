#!/bin/sh
#
# encode '.wav' to '.flac' with tags
#
# INFO_WAV_LIST format:
# +--------------------------
# |# ARTIST=artist
# |# ALBUM=album
# |# DATE=date (YYYY-MM-DD)
# |# GENRE=genre
# |
# |[TRACKNUMBER] - TITLE.wav
# |[TRACKNUMBER] - TITLE.wav
# |...
# +--------------------------
#
# Ref:
# FLAC: https://xiph.org/flac/
# VorbisComment: http://wiki.xiph.org/VorbisComment
# Vorbis Comment Field Recommendations:
#     http://age.hobba.nl/audio/mirroredpages/ogg-tagging.html
#
#
# 2014-01-24
# LiweitiaNux
#

if ! which flac > /dev/null 2>&1; then
    echo "ERROR: 'flac' not found!"
    exit 1
fi

if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "    `basename $0` <info_wav_list>"
    exit 2
fi

INFO="$1"

## meta data
# ARTIST
if grep -iq '^#.*ARTIST' $INFO; then
    ARTIST=`grep -i '^#.*ARTIST' $INFO | sed 's/^#.*ARTIST=//'`
else
    ARTIST=""
fi
# ALBUM
if grep -iq '^#.*ALBUM' $INFO; then
    ALBUM=`grep -i '^#.*ALBUM' $INFO | sed 's/^#.*ALBUM=//'`
else
    ALBUM=""
fi
# DATE
if grep -iq '^#.*DATE' $INFO; then
    DATE=`grep -i '^#.*DATE' $INFO | sed 's/^#.*DATE=//'`
else
    DATE=""
fi
# GENRE
if grep -iq '^#.*GENRE' $INFO; then
    GENRE=`grep -i '^#.*GENRE' $INFO | sed 's/^#.*GENRE=//'`
else
    GENRE=""
fi
# PUBLISHER
if grep -iq '^#.*PUBLISHER' $INFO; then
    PUBLISHER=`grep -i '^#.*PUBLISHER' $INFO | sed 's/^#.*PUBLISHER=//'`
else
    PUBLISHER=""
fi

echo "ARTIST: $ARTIST"
echo "ALBUM: $ALBUM"
echo "DATE: $DATE"
echo "GENRE: $GENRE"
echo "PUBLISHER: $PUBLISHER"

## process wav file list
grep '^[/a-zA-Z0-9].*' $INFO | while read f; do
    file=`basename "$f"`
    TRACKNUMBER=`echo "$file" | sed 's/\ -\ .*\.wav$//'`
    TITLE=`echo "$file" | sed -e 's/^[0-9]*\ -\ //' -e 's/\.wav$//'`
    echo "TRACKNUMBER: $TRACKNUMBER; TITLE: $TITLE ..."
    # encode
    flac -T "ARTIST=${ARTIST}" -T "TITLE=${TITLE}" \
        -T "ALBUM=${ALBUM}" -T "DATE=${DATE}" \
        -T "TRACKNUMBER=${TRACKNUMBER}" \
        -T "GENRE=${GENRE}" -T "PUBLISHER=${PUBLISHER}" \
        -s "${f}"
    if [ $? -eq 0 ]; then
        echo "    OK"
    else
        echo "    *** ERROR ***"
    fi
done

