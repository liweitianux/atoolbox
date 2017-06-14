#!/bin/sh
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT License
#
# Manually control NVIDIA GPU fan speed using `nvidia-settings` tool.
#
# NOTE:
# Make sure the NVIDIA persistence daemon `nvidia-persistenced` is
# running to keep the GPU context.  Otherwise, the applied settings
# will be reverted after X exited.
#
# NOTE:
# This script should work on Debian Linux (testing; amd64)
#
# Credits:
# [1] NVIDIA GPU Coolness
#     https://sites.google.com/site/akohlmey/random-hacks/nvidia-gpu-coolness
# [2] http://klein-group.icms.temple.edu/akohlmey/files/set-gpu-fans.tar.gz
#
#
# 2017-06-14
#

set -e

DEBUG=

error() {
    echo "ERROR: $@" >&2
}

debug() {
    if [ -n "${DEBUG}" ]; then
        echo "DEBUG: $@" >&2
    fi
}


make_edid() {
    # Generate this with: base64 <file>
    base64 -d > "$1" <<_EOF_
AP///////wAQrBXwTFRBNS4TAQOANCB47h7Frk80sSYOUFSlSwCBgKlA0QBxTwEBAQEBAQEBKDyA
oHCwI0AwIDYABkQhAAAaAAAA/wBDNTkyTTlCOTVBVEwKAAAA/ABERUxMIFUyNDEwCiAgAAAA/QA4
TB5REQAKICAgICAgAB0=
_EOF_
}


make_xorgconf() {
    local xcfg busid edidfile
    xcfg="$1"
    busid="$2"
    edidfile="$3"
    cat > "${xcfg}" <<_EOF_
# Simple single screen X server config with fake CRT attached

Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"     0    0
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "VideoCard0"
    Monitor        "Monitor0"
    DefaultDepth   24
    Option         "UseDisplayDevice" "DFP-0"
    Option         "ConnectedMonitor" "DFP-0"
    Option         "CustomEDID" "DFP-0:${edidfile}"
    Option         "Coolbits" "5"
    SubSection "Display"
        Depth   24
        Modes   "1024x768"
    EndSubSection
EndSection

Section "Files"
    ModulePath   "/usr/lib/xorg/modules/extensions/nvidia"
    ModulePath   "/usr/lib/xorg/modules"
EndSection

Section "ServerFlags"
    Option   "AllowEmptyInput" "on"
    Option   "Xinerama"        "off"
    Option   "SELinux"         "off"
EndSection

Section "Device"
    Identifier   "VideoCard0"
    Driver       "nvidia"
    Screen       0
    Option       "UseDisplayDevice" "DFP-0"
    Option       "ConnectedMonitor" "DFP-0"
    Option       "CustomEDID" "DFP-0:${edidfile}"
    Option       "Coolbits" "4"
    BusID        "${busid}"
EndSection

Section "Monitor"
    Identifier   "Monitor0"
    Vendorname   "Dummy Display"
    Modelname    "1024x768"
EndSection
_EOF_
}


# Convert the PCI ID from format "<bus>:<device>.<func>" to
# "PCI:<bus>:<device>:<func>" for Xorg usage.
make_busid() {
    local busid cmd
    busid=$(echo "$1" | \
            sed -e 's/\([0-f]\+\):\([0-f]\+\)\.\([0-f]\+\)/0x\1 0x\2 0x\3/')
    debug make_busid:busid: ${busid}
    cmd="printf 'PCI:%d:%d:%d' ${busid}"
    debug make_busid:cmd: ${cmd}
    busid=$(eval ${cmd})
    debug make_busid:busid: ${busid}
    echo ${busid}
}


get_gpu_busid() {
    local busid=""
    for id in $(lspci | grep "VGA compatible controller: NVIDIA" | \
                cut -d' ' -f1); do
        debug get_gpu_busid:id: ${id}
        busid="${busid} $(make_busid ${id})"
    done
    debug get_gpu_busid:busid: ${busid}
    echo "${busid}"
}


# Default display number for Xorg
XD=:0
# Default fan speed (70%)
FANSPEED=70

usage() {
    local progname
    progname="${0##*/}"
    cat <<_EOF_
usage:
    ${progname} start [ speed_percent ]
        Set the GPU fan speed to given percent (default: ${FANSPEED})
    ${progname} stop
        Go back to automatic fan control
_EOF_
}

case "$1" in
    start)
        shift
        FANSPEED=${1:-${FANSPEED}}
        CMD="nvidia-settings -a [gpu:0]/GPUFanControlState=1 \
                             -a [fan:0]/GPUTargetFanSpeed=${FANSPEED}"
        ;;
    stop)
        CMD="nvidia-settings -a [gpu:0]/GPUFanControlState=0"
        ;;
    *)
        usage
        exit 1
        ;;
esac

EDIDFILE=$(mktemp /tmp/edid-XXXXXXXX.bin)
make_edid ${EDIDFILE}

BUSID=$(get_gpu_busid)
if [ -z "${BUSID}" ]; then
    error "no NVIDIA GPU found!" >&2
    exit 2
fi

if [ $(id -u) -ne 0 ]; then
    error "root privileges required!" >&2
    exit 3
fi

for busid in ${BUSID}; do
    echo "Set fan speed for NVIDIA GPU: ${busid} ..."
    XCFG=$(mktemp /tmp/xorg-XXXXXXXX.conf)
    make_xorgconf ${XCFG} ${busid} ${EDIDFILE}
    Xorg ${XD} -config ${XCFG} -keeptty >/dev/null &
    XPID=$!
    sleep 3
    echo "Started Xorg at display ${XD} with PID=${XPID}."
    env DISPLAY=${XD} ${CMD}
    sleep 1
    kill -9 ${XPID}
    echo "Killed Xorg PID=${XPID}."
    rm -f ${XCFG}
done

rm -f ${EDIDFILE}

exit 0
