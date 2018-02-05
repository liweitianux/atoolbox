#!/bin/sh
#
# Find the PID of a command in UNIX
# Jon Cairns
# http://blog.joncairns.com/2013/03/finding-the-process-id-pid-of-a-command-in-unix/
# 2018-03-20
#

me="${0##*/}"
if [ $# -lt 1 ]; then
    echo "$me: a grep expression for filtering processes required" >&2
    exit 1
fi

output=$(ps auxww | grep "$*" | grep -v grep | grep -v $0)
lines=$(echo "$output" | wc -l)

if [ $lines -gt 1 ]; then
    echo "$me: multiple processes matching the expression: '$*'" >&2
    echo
    echo "$output" >&2
    exit 2
elif [ -z "$output" ]; then
    echo "$me: no processes matching the expression: '$*'" >&2
    exit 3
fi

echo "$output" | awk '{ print $2 }'
