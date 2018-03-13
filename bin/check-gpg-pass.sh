#!/bin/sh
#
# Copyright (c) 2018 Aaron LI <aly@aaronly.me>
# MIT License
#
# `gpg-agent` is required by `gpg` to manage the private keys.  It also
# caches the passphrase of the private keys, and will ask user if the
# cached passphrase expired, which may be annoying for the cron tasks.
#
# Workaround the `gpg-agent` asking for passphrase by testing its current
# status about the cached passphrase, i.e., with *pinentry* disabled
# (by passing the `--pinentry-mode error` to `gpg`), invoke `gpg` to try
# to sign a message, if it succeed, then the necessary passphrase has
# already been cached by `gpg-agent`.
#
# Reference:
# * Programmatically determine if gpg-agent will ask for passphrase
#   https://superuser.com/a/1212720/731908
#
# Aaron LI
# 2018-03-13
#

# Workaround to make `notify-send` work with cron
# Credit: https://stackoverflow.com/a/16520076
export DISPLAY=:0
export XAUTHORITY="${HOME}/.Xauthority"

PROG="${0##*/}"
# Command to send notification
NOTIFY_CMD="notify-send"


error() {
    echo "$*" >&2
}

exists() {
    command -v "$1" >/dev/null 2>&1
}

notify() {
    local message="$1"
    if exists ${NOTIFY_CMD}; then
        command ${NOTIFY_CMD} ${PROG} "${message}"
    fi
}


# Check whether `gpg-agent` already cached the needed passphrase,
# if return 0 (i.e., success), then the passphrase is already cached,
# then `pass` can decrypt the password without triggering `gpg-agent`
# to ask user for the passphrase.
check_cached_passphrase() {
    local key
    [ -n "$1" ] && key="--local-user $1" || key=""
    echo "test" | \
        gpg --sign --batch --no-tty --pinentry-mode error \
            ${key} -o /dev/null >/dev/null 2>&1
}


case "$1" in
    -h|--help)
        echo "usage: ${PROG} [keyname]"
        exit 1
        ;;
esac

check_cached_passphrase "$1"
rv=$?
if [ ${rv} -ne 0 ]; then
    msg="GPG passphrase not cached!"
    error "${msg}"
    notify "${msg}"
fi
exit ${rv}
