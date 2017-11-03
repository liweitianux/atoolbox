#!/usr/bin/env bash
#
# Command line – Let it snow, Let it snow, Let it snow
# http://mebsd.com/coding-snipits/command-line-let-it-snow-let-it-snow-let-it-snow.html
# October 9, 2012
#

LINES=${LINES:-24}
COLUMNS=${COLUMNS:-90}

clear; \
while :; \
do \
echo $LINES $COLUMNS $(($RANDOM%$COLUMNS)); \
sleep 0.1; \
done | \
awk '{ \
         a[$3]=0; \
         for(x in a) {o=a[x]; \
                      a[x]=a[x]+1; \
                      printf "\033[%s;%sH ",o,x; \
                      printf "\033[%s;%sH*\033[0;0H",a[x],x; \
                     } \
     }'
