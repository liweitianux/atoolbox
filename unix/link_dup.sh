#!/bin/sh
#
# Replace duplicate files with symbolic links.
#
# <dup_list> format (dup* will be removed and linked to FILE):
# FILE dup1 dup2 ...
#
# NOTE:
# * Filename cannot contain spaces.
#
# Weitian LI
# 2015/03/06
#

err_msg() {
    echo "$@" > /dev/stderr
}

usage() {
    err_msg "Usage:"
    err_msg "    $1 <dup_list>"
}

md5() {
    file="$1"
    os=`uname -s`
    case "${os}" in
        Linux)
            MD5=`md5sum ${file} | awk '{ print $1 }'`
            ;;
        FreeBSD)
            MD5=`md5 ${file} | awk '{ print $NF }'`
            ;;
        *)
            err_msg "ERROR: Unsupported OS: ${os}"
            exit 255
    esac
    echo ${MD5}
}

mk_link_args() {
    # Convert a dup_line of
    # 'FILE dup1 dup2 ...'
    # into format of
    # 'FILE dup1'
    # 'FILE dup2'
    # '...'
    # for use in link
    echo "$@" | awk '{
        for (i=2; i<=NF; i++) {
            printf("%s %s\n", $1, $i);
        }
    }'
}

md5_link() {
    # Link file if its md5 is equal to the original FILE.
    # Args: 'FILE dup'
    file="$1"
    dup="$2"
    file_path=`readlink -f "${file}"`
    dup_path=`readlink -f "${dup}"`
    if [ "${file_path}" = "${dup_path}" ]; then
        err_msg "WARNING: '${file}' and '${dup}' are the same file"
    else
        md5_file=`md5 ${file}`
        md5_dup=`md5 ${dup}`
        if [ "${md5_file}" = "${md5_dup}" ]; then
            #err_msg "rm -v \"${dup}\""
            #err_msg "ln -sv \"${file}\" \"${dup}\""
            rm -v "${dup}"
            ln -sv "${file}" "${dup}"
        else
            err_msg "WARNING: '${file}' different to '${dup}'"
        fi
    fi
}


if [ $# -ne 1 ]; then
    usage `basename $0`
    exit 1
fi

dup_list="$1"
cat "${dup_list}" | while read line; do
    err_msg "${line}"
    mk_link_args "${line}" | while read lk; do
        err_msg " => " ${lk}
        md5_link ${lk}
    done
done

