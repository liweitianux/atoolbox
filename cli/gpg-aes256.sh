#!/bin/sh
#
# Copyright (c) 2015,2018 Aaron LI
#
# Encrypt a file using GPG with symmetric AES256 algorithm.
#

case "$1" in
    ""|-h|--help)
        echo "usage: ${0##*/} <infile>"
        exit 1
        ;;
esac

if which gpg2 >/dev/null 2>&1; then
    GPG=gpg2
elif which gpg >/dev/null 2>&1; then
    GPG=gpg
else
    echo "ERROR: cannot find gpg/gpg2!"
    exit 2
fi

infile="$1"
outfile="${infile}.gpg"
${GPG} --symmetric --cipher-algo aes256 -o "${outfile}" "${infile}"
chmod 0600 "${outfile}"
echo "Encrypted file saved to: ${outfile}"
