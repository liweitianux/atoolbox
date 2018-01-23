#!/bin/sh
#
# Start the `FRP client' as a daemon, and start/check with a cron job.
# https://github.com/fatedier/frp
#
# See start-stop-daemon(8)
#
# Example configuration file:
# ---------------------------------------------------------------------
# [common]
# server_addr = xxx.xxx.xxx.xxx
# server_port = xxxx
#
# [ssh]
# type = tcp
# local_ip = 127.0.0.1
# local_port = 22
# # port on the remote server that will be proxied to here
# # e.g., ssh -p <remote_port> <user>@<server_addr>
# remote_port = 2334
# ---------------------------------------------------------------------
#
# Example cron job (check every 10th minutes):
# */10  *  *  *  *  sh $HOME/bin/frpc-daemon.sh start >/dev/null 2>&1
#
#
# Aaron LI
# 2018-01-23
#

NAME=frpc
PROGRAM=$HOME/local/frp/frpc
CONFIG=$HOME/configs/${NAME}.ini
PIDFILE=$HOME/.cache/${NAME}.pid
LOGFILE=$HOME/.cache/${NAME}.log

PATH=${PATH}:/usr/local/sbin:/usr/sbin:/sbin

case "$1" in
    start)
        cmd=start
        cmd_args="--start --make-pidfile --background"
        ;;
    stop)
        cmd=stop
        cmd_args="--stop --remove-pidfile --retry 5"
        ;;
    status)
        cmd=status
        cmd_args="--status"
        ;;
    *)
        echo "usage: ${0##*/} {start|stop|status}"
        exit 1
        ;;
esac

start-stop-daemon ${cmd_args} \
    --oknodo --verbose \
    --name ${NAME} \
    --pidfile ${PIDFILE} \
    --startas ${PROGRAM} \
    -- -c ${CONFIG} -L ${LOGFILE}
ret=$?
if [ "${cmd}" = "start" ]; then
    if [ ${ret} -eq 0 ]; then
        echo "Successfully started ${NAME} as PID $(cat ${PIDFILE})"
    else
        echo "Failed to start ${NAME}: ${PROGRAM}"
        echo "Exit code: ${ret}"
    fi
elif [ "${cmd}" = "stop" ]; then
    if [ ${ret} -eq 0 ]; then
        echo "Successfully stopped ${NAME}: ${PROGRAM}"
    else
        echo "Failed to stop ${NAME}: ${PROGRAM}"
        echo "Exit code: ${ret}"
    fi
elif [ "${cmd}" = "status" ]; then
    if [ ${ret} -eq 0 ]; then
        echo "${NAME} is running as PID $(cat ${PIDFILE})"
    elif [ ${ret} -eq 1 ]; then
        echo "${NAME} is not running but the PID file (${PIDFILE}) exists"
    elif [ ${ret} -eq 3 ]; then
        echo "${NAME} is not running"
    else
        echo "${NAME} in unknown status"
        echo "Exit code: ${ret}"
    fi
else
    echo "ERROR: unknown command: ${cmd}"
fi
