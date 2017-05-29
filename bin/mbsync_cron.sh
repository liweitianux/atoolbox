#!/bin/sh
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT license
#
# Get new mails and sync IMAP folders using `mbsync` with `cron`.
#
# Since the passwords are managed using `pass` which encrypts them using
# `gpg`, therefore, `gpg-agent` will be periodically triggered to ask user
# for the passphrase to decrypt the stored passwords, which is annoying
# for the cron tasks.
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
# 2017-05-28
#

set -e

# Workaround to make `notify-send` work with cron
# Credit: https://stackoverflow.com/a/16520076
export DISPLAY=:0
export XAUTHORITY="${HOME}/.Xauthority"

PROG="${0##*/}"

LOGFILE="/dev/null"

# The GPG key used by `pass`
GPG_KEY="aly@aaronly.me"

# Command to send notification
NOTIFY_CMD="notify-send"


error() {
    echo "$*" >&2
}

log() {
    echo "[$(date --iso-8601=seconds)] $*" >> ${LOGFILE}
}

exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a process id (PID) exists
# Credit: https://stackoverflow.com/a/3044045
is_running() {
    kill -0 "$1" >/dev/null 2>&1
}

notify() {
    local message="$1"
    if exists ${NOTIFY_CMD}; then
        command ${NOTIFY_CMD} ${PROG} "${message}"
    fi
}

usage() {
    error "usage: ${PROG} <account>"
}


# Check whether `gpg-agent` already cached the needed passphrase,
# if return 0 (i.e., success), then the passphrase is already cached,
# then `pass` can decrypt the password without triggering `gpg-agent`
# to ask user for the passphrase.
check_cached_passphrase() {
    local key="$1"
    echo "test" | \
        gpg2 --sign --batch --no-tty --pinentry-mode error \
             --local-user ${key} -o /dev/null >/dev/null 2>&1
}


# Check the network availability and remote server online status.
# FIXME: need test IMAP connection instead of ping the server...
check_server() {
    local server="$1"
    ping -c 3 ${server} >/dev/null 2>&1
}


case "$1" in
    aly|weitian|webmaster)
        ACCOUNT="$1"
        SERVER="mail.aaronly.me"
        ;;
    autistici)
        ACCOUNT="$1"
        SERVER="mail.autistici.org"
        ;;
    lavabit)
        ACCOUNT="$1"
        SERVER="imap.lavabit.com"
        ;;
    outlook-aly|outlook-li)
        ACCOUNT="$1"
        SERVER="imap-mail.outlook.com"
        ;;
    gmail-aly|gmail-li)
        ACCOUNT="$1"
        SERVER="imap.gmail.com"
        ;;
    sjtu)
        ACCOUNT="$1"
        SERVER="imap.sjtu.edu.cn"
        ;;
    foxmail)
        ACCOUNT="$1"
        SERVER="imap.qq.com"
        ;;
    -h|--help)
        usage
        exit 1
        ;;
    *)
        error "Invalid account name!"
        usage
        exit 1
        ;;
esac

# PID file for `mbsync`, which may just hang and lock the mailbox
PIDFILE="${HOME}/.cache/mbsync.${ACCOUNT}.pid"

# Log file to record the sync information
LOGFILE="${HOME}/.cache/mbsync.${ACCOUNT}.log"
[ ! -f ${LOGFILE} ] && touch ${LOGFILE}
log "-----------------------------------"


if ! check_cached_passphrase ${GPG_KEY}; then
    log "GPG passphrase not cached! Skip sync mail."
    notify "GPG passphrase not cached! Skip sync mail."
    exit 2
fi

#if ! check_server ${SERVER}; then
#    log "Network not available or remote server ${SERVER} is offline!"
#    notify "Network not available or remote server ${SERVER} is offline!"
#    exit 3
#fi

if [ -f "${PIDFILE}" ]; then
    PID=$(cat ${PIDFILE})
    if is_running ${PID} && ! kill ${PID} >/dev/null 2>&1; then
        log "Cannot kill process: ${PID}"
        notify "Cannot kill process: ${PID}"
    else
        log "Killed previous instance: ${PID}"
        notify "Killed previous instance: ${PID}"
    fi
fi

log "Sync mail for account: ${ACCOUNT} ..."
notify "Sync mail for account: ${ACCOUNT} ..."
mbsync -V "${ACCOUNT}" >>${LOGFILE} 2>&1 &
#sleep 10 &
PID=$!
echo ${PID} > ${PIDFILE}
wait ${PID}
rm ${PIDFILE}

log "Finish syncing mail for account: ${ACCOUNT}!"
notify "Finish syncing mail for account: ${ACCOUNT}!"
