#!/bin/sh
#
# Copyright (c) 2019 Aaron LI
# MIT License
#
# Generate random passwords.
#
# Credit:
# * How to generate a random string?
#   https://unix.stackexchange.com/a/230676
# * password-store:
#   http://www.passwordstore.org/
#

# usage: generate ${characters} ${length}
generate() {
    local characters="$1"
    local length="$2"
    env LC_ALL=C tr -dc "${characters}" </dev/urandom |
        head -c ${length}
}

usage() {
    cat << _EOF_
Generate random passwords.

usage:
    ${0##*/} [-n] [-l length] [count]

options:
    -n : no symbols, i.e., only alphabets and numbers
    -l length : password length (default: 16)
    count : number of passwords to generate (default: 1)
_EOF_

    exit 1
}

main() {
    local characters="[:graph:]"
    local length=16
    local count=1

    while getopts :hl:n opt; do
        case ${opt} in
        h)
            usage
            ;;
        l)
            length="${OPTARG}"
            ;;
        n)
            characters="[:alnum:]"
            ;;
        \?)
            echo "Invalid option -${OPTARG}"
            usage
            ;;
        :)
            echo "Option -${OPTARG} requires an argument"
            usage
            ;;
        esac
    done

    shift $((${OPTIND} - 1))
    if [ $# -eq 0 ]; then
        :
    elif [ $# -eq 1 ]; then
        count="$1"
    else
        usage
    fi

    for i in $(seq ${count}); do
        generate "${characters}" ${length}
        echo
    done
}

main "$@"
