#!/bin/sh
#
# Get the public IPv4 & IPv6 address.
#
# Credit:
# * icanhaz:
#   - https://icanhazip.com
#   - https://major.io/icanhazip-com-faq/
#   - https://github.com/major/icanhaz
# * http://ifconfig.me/
# * http://ipecho.net/
#
# Aaron LI
# 2017-08-31
# MIT license
#

#URL="http://ifconfig.me"  # IPv4 only
#URL="http://ipecho.net/plain"  # IPv4 only
URL="https://icanhazip.com"  # IPv4 & IPv6

ipv4=$(curl -4s ${URL} 2>/dev/null)
ipv6=$(curl -6s ${URL} 2>/dev/null)
[ -n "${ipv4}" ] && echo "IPv4: ${ipv4}"
[ -n "${ipv6}" ] && echo "IPv6: ${ipv6}"
