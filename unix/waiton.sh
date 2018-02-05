#!/bin/sh
#
# Wait for a UNIX process to finish
# Jon Cairns
# http://blog.joncairns.com/2013/03/wait-for-a-unix-process-to-finish/
# 2013-03-20
#

case "$1" in
    -h|--help|"")
        echo "usage: ${0##*/} <pid>" >&2
        exit 1
        ;;
esac

pid="$1"
me="${0##*/}($$)"
name=$(ps -p ${pid} -o comm=)
if [ $? -eq 0 ]; then
    echo "${me}: waiting for process (${pid}, ${name}) to finish ..."
    while ps -p ${pid} >/dev/null 2>&1; do
        echo -n .
        sleep 1
    done
    echo
else
    echo "${me}: failed to find process with PID ${pid}" >&2
    exit 2
fi
