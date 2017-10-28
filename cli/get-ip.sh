#!/bin/sh
#
# Get the public IPv4 address.
#
# Credit:
# * http://ifconfig.me/
# * http://ipecho.net/
#
# Aaron LI
# 2017-08-31
# MIT license
#

URL="http://ifconfig.me"
URL="http://ipecho.net/plain"

ip=$(curl -s ${URL})
echo ${ip}
