#!/bin/sh
#
# Extract a zip archive and fix Chinese filenames.
#
# Credit: https://superuser.com/a/872616
#

has() {
	type "$1" >/dev/null 2>&1
}

extract() {
	if has bsdtar; then
		# bsdtar provided by libarchive
		bsdtar -xvf "$1"
	elif has 7z; then
		# 7z provided by p7zip
		env LC_ALL=C 7z x "$1"
	else
		echo "ERROR: Neither bsdtar nor 7z found" >&2
		exit 1
	fi
}

fixnames() {
	find . -depth | while read -r p; do
		dn=$(dirname "${p}")
		fn=$(basename "${p}")
		fn2=$(echo "${fn}" | iconv -f gbk -t utf-8)
		if [ "${fn}" != "${fn2}" ]; then
			mv -v "${dn}/${fn}" "${dn}/${fn2}"
		fi
	done
}

case $1 in
'' | -h | --help)
	echo "usage: ${0##*/} <zip>"
	exit 1
	;;
esac

zipfile=$(realpath "$1")
curdir=$(pwd)
tmpdir=$(mktemp -d)
cd "${tmpdir}"

echo "Extracting archive '${zipfile}' ..."
extract "${zipfile}"
echo "Fixing filenames ..."
fixnames

cd "${curdir}"
mv "${tmpdir}"/* .
rmdir "${tmpdir}"
