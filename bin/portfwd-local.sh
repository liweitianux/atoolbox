#!/bin/sh
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT license
#
# Local port fowarding using SSH
#
# Credits:
# [1] https://help.ubuntu.com/community/SSH/OpenSSH/PortForwarding
# [2] https://unix.stackexchange.com/a/115906
#
# Aaron LI
# 2017-01-11
#

if [ $# -ne 3 ]; then
    echo "Usage:"
    echo "    `basename $0` <local_port> <dest_port> <host>"
    exit 1
fi

LPORT="$1"
DPORT="$2"
HOST="$3"

ssh -L ${LPORT}:localhost:${DPORT} ${HOST}
