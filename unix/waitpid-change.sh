#!/bin/sh
#
# Find the PID of a command and wait until it changes.
#
# Aaron LI
# 2018-04-01
#

self="$0"

getpid() {
    pattern="$1"
    output=$(ps auxww | grep "${pattern}" | grep -v grep | grep -v ${self})
    lines=$(echo "${output}" | wc -l)

    if [ ${lines} -gt 1 ]; then
        echo "Error: multiple processes matching the pattern: '${pattern}'" >&2
        echo
        echo "${output}" >&2
        exit 2
    elif [ -z "${output}" ]; then
        echo "Error: no processes matching the pattern: '${pattern}'" >&2
        exit 3
    fi

    echo "${output}" | awk '{ print $2 }'
}

usage() {
    echo "usage: ${self##*/} [-h] [-d delay] <pattern>"
    exit 1
}

while getopts :d:h opt; do
    case "${opt}" in
        d)
            delay=${OPTARG}
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option -${OPTARG}" >&2
            usage
            ;;
        :)
            echo "Option -${OPTARG} requires an argument" >&2
            usage
            ;;
    esac
done
shift $((OPTIND - 1))
[ $# -ne 1 ] && usage

delay=${delay:-5}  # [second]
pattern="$1"

pid=$(getpid "${pattern}") || exit $?
echo "Process ${pid} matches pattern: '${pattern}'"
echo -n "waiting for pid change ..."
sleep ${delay}
while pid2=$(getpid "${pattern}"); do
    [ ${pid2} -ne ${pid} ] && {
        echo "changed!"; exit 0
    }
    echo -n "."
    sleep ${delay}
done
exit $?
