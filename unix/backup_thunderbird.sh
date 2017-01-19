#!/bin/sh
#
# Backup thunderbird data with rsync.
#
# Weitian LI <liweitianux@gmail.com>
# 2015/01/09
#

SCRIPT_PATH=`readlink -f $0`
# Backup destination: same directory as this script.
DEST=`dirname ${SCRIPT_PATH}`

# rsync options
RSYNC_OPTS="-az"

VERBOSE=FALSE

usage() {
    echo "Usage:"
    echo "`basename $0` [ -hDv ] [ -d <dest_dir> ] <thunderbird_profile_dir>"
    echo ""
    echo "    -h: show this help"
    echo "    -D: allow delete destination files"
    echo "    -v: show verbose information"
    echo ""
    echo "This script backups thunderbird mails and data with rsync."
    echo ""
}

# disable 'verbose error handling' by preceeding with a colon(:)
while getopts ":hDvd:" opt; do
    case $opt in
        h)
            usage
            exit 0
            ;;
        D)
            RSYNC_OPTS="${RSYNC_OPTS} --delete"
            ;;
        v)
            RSYNC_OPTS="${RSYNC_OPTS} -v"
            VERBOSE=TRUE
            ;;
        d)
            DEST="${OPTARG}"
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            exit 2
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            exit 3
            ;;
    esac
done

# shift the options processed by getopts
shift $((${OPTIND} - 1))

if [ $# -ne 1 ]; then
    usage
    exit 1
fi

# the remaining argument after shift
SRC="$1"

if [ "${VERBOSE}" = "TRUE" ]; then
    echo "RSYNC_OPTS: ${RSYNC_OPTS}"
    echo "SRC: ${SRC}"
    echo "DEST: ${DEST}"
fi

# backup files and directories
BACKUP_LIST="ImapMail/"                 # IMAP mail boxes
BACKUP_LIST="${BACKUP_LIST} Mail/"      # POP & Local mail boxes
BACKUP_LIST="${BACKUP_LIST} abook.mab history.mab"  # personal & collected addresses
BACKUP_LIST="${BACKUP_LIST} persdict.dat"   # personal spelling dictionary
BACKUP_LIST="${BACKUP_LIST} prefs.js"   # preferences & tags definitions
BACKUP_LIST="${BACKUP_LIST} key3.db signons.sqlite cert8.db"    # saved passwords
#BACKUP_LIST="${BACKUP_LIST} cookies.sqlite permissions.sqlite storage.sdb" # Lightning add-on

# check files and directories; and rsync
for i in ${BACKUP_LIST}; do
    if [ -e "${SRC}/${i}" ]; then
        CMD="rsync ${RSYNC_OPTS} '${SRC}/${i}' '${DEST}/${i}'"
        if [ "${VERBOSE}" = "TRUE" ]; then
            echo "CMD: ${CMD}"
        fi
        eval ${CMD}
    else
        echo "${SRC}/${i}: not exist!" >&2
    fi
done

# log
DATETIME=`date -u +%FT%TZ`
echo ""
echo "Thunderbird data sync finished at ${DATETIME}"
echo "${DATETIME}" >> "${DEST}/SYNC.log"

