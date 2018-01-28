#!/bin/sh
#
# Encrypt the given file using symmetric AES256 algorithm.
#
# Aaron LI
# 2015-01-24
#

case "$1" in
    ""|-h|--help)
        echo "usage: ${0##*/} <infile>"
        exit 1
        ;;
esac

infile="$1"
outfile="${infile}.gpg"
gpg2 --symmetric --cipher-algo aes256 -o "${outfile}" "${infile}"
chmod 0600 "${outfile}"
