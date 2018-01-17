#!/bin/sh
#
# Convert text file to UTF-8 encoding.
#
# Aaron LI
# 2018-01-17
#

case "$1" in
    ""|-h|--help)
        echo "usage: ${0##*/} <file1> ..."
        exit 1;
        ;;
esac

convert() {
    from="$1"
    infile="$2"
    iconv -f $from -t UTF-8 "$infile" > _tmp.$$
    if [ $? -eq 0 ]; then
        mv _tmp.$$ "$infile"
        echo "converted to UTF-8 from $from"
    else
        rm _tmp.$$
        echo "ERROR: failed conversion from $from"
    fi
}

while [ -n "$1" ]; do
    infile="$1"
    shift;
    echo "=== $infile ==="
    if file "$infile" | grep -q "UTF-8 Unicode"; then
        echo "already UTF-8; skip it"
    elif file "$infile" | grep -q "UTF-16 Unicode"; then
        convert UTF-16 "$infile"
    else
        convert GBK "$infile"
    fi
done
