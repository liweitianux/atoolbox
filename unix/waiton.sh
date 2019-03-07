#!/bin/sh
#
# Copyright (c) 2018-2019 Aaron LI <aly@aaronly.me>
# MIT License
#
# Wait for process(s) to finish ...
#
# Credit: http://blog.joncairns.com/2013/03/wait-for-a-unix-process-to-finish/
#

me="${0##*/}"

has_proc() {
    ps -p $1 >/dev/null 2>&1
}

get_proc_cmd() {
    if has_proc $1; then
        ps -p $1 -o args=
    else
        echo "(not found)"
        return 1
    fi
}

usage() {
    cat <<_EOF_

Wait for process(s) to finish ...

usage:
    ${me} [-t interval] <pid1> [<pid2> ...]

options:
    -t : number of seconds between every check.
         (default: 5)

example:
    ${me} 123 456 && command ...

_EOF_
    exit 1
}


interval=5
while getopts :ht: opt; do
    case ${opt} in
    h)
        usage
        ;;
    t)
        interval="${OPTARG}"
        ;;
    \?)
        echo "${me}: invalid option -${OPTARG}"
        usage
        ;;
    :)
        echo "${me}: option -${OPTARG} requires an argument"
        usage
        ;;
    esac
done

shift $((${OPTIND} - 1))
[ $# -eq 0 ] && usage

echo "${me}: waiting for following process(s) to finish ..."
pids="$@"
nwait=0
for p in ${pids}; do
    cmd=$(get_proc_cmd ${p})
    ret=$?
    echo "* [pid: ${p}] ${cmd}"
    if [ ${ret} -eq 0 ]; then
        nwait=$((${nwait} + 1))
        eval _has_${p}=yes
    fi
done

echo -n "${me}: "
while [ ${nwait} -ne 0 ]; do
    nwait=0
    for p in ${pids}; do
        key="_has_${p}"
        eval last=\$${key}
        if has_proc ${p}; then
            nwait=$((${nwait} + 1))
            eval ${key}=yes
        elif [ -n "${last}" ]; then
            echo
            echo "${me}: pid=${p} exited."
            echo -n "${me}: "
            eval ${key}=
        fi
    done
    echo -n "${nwait}."
    sleep ${interval}
done

echo "done"
exit 0
