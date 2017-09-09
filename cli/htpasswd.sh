#!/bin/sh
#
# Generate htpasswd (Apache password file) without installing Apache tools.
#
# Credit:
# * NGINX: FAQ
#   https://www.nginx.com/resources/wiki/community/faq/
# * How to get a password from a shell script without echoing
#   https://stackoverflow.com/a/3980713/4856091
#
#
# Aaron LI
# 2017-09-09
#

if [ $# -ne 1 ]; then
    echo "usage: ${0##*/} <username>"
    exit 1
fi

USER="$1"

# Disable echo ...
stty -echo
printf "Password: "
read PASS1
printf "\n"
printf "Password (again): "
read PASS2
printf "\n"
# Enable echo
stty echo

if [ "${PASS1}" != "${PASS2}" ]; then
    echo "ERROR: passwords do not match!"
    exit 2
fi

# Use the UNIX "crypt" encryption
printf "${USER}:$(openssl passwd -crypt ${PASS1})\n"
