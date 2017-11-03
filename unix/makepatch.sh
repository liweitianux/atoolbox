#!/bin/sh
#
# Automatically make the patches for the updated files by find
# the corresponding '*.orig' files through the given directory.
#
# Based on the 'makepatch' target of FreeBSD 'Mk/bsd.port.mk'.
#
#
# Aaron LI
# Created: 2015-10-13
# Updated: 2015-12-07
#
# ChangeLog:
# 2015-12-06:
#   * Drop option '-s' for `find`
# 2015-12-07:
#   * Replace 'workdir' with 'srcdir'
#   * Make find to search 'srcdir' instead of '.'
#

case "$1" in
    -[hH]*|--[hH]*)
        echo "Usage: $0 [srcdir] [patchdir]"
        echo ""
        echo "Both 'srcdir' and 'patchdir' default to be ."
        exit 1
        ;;
esac

SRCDIR=${1:-.}
PATCHDIR=${2:-.}

if [ ! -d "${PATCHDIR}" ]; then
    mkdir ${PATCHDIR}
fi

PATCH_PATH_SEPARATOR="_"

for f in `find ${SRCDIR}/ -type f -name '*.orig'`; do
    _SEEN="NO"
    ORIG=${f#./}
    NEW=${ORIG%.orig}
    if cmp -s ${ORIG} ${NEW}; then
        # The two files are identical
        continue
    fi
    PATCH=`echo ${NEW} | sed -e \
        "s|${PATCH_PATH_SEPARATOR}|&&|g" -e "s|/|${PATCH_PATH_SEPARATOR}|g"`
    OUT="${PATCHDIR}/patch-${PATCH}"
    if test -f "${OUT}"; then
        echo WARNING: patch already exists: \"${OUT}\"
    else
        echo diff -udp ${ORIG} ${NEW} '>' ${OUT}
        TZ=UTC diff -udp ${ORIG} ${NEW} | sed -e \
            '/^---/s|\.[0-9]* +0000$| UTC|' -e \
            '/^+++/s|\([[:blank:]][-0-9:.+]*\)*$||' > ${OUT}
    fi
done

exit 0

