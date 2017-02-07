#!/bin/sh
#
# Unlock the user sessions when the screen locker failed/crashed
# (e.g., during an upgrade).
#
# This is the solution for ConsoleKit (e.g., on Gentoo Linux).
# As for the `systemd` case, simply run:
#    $ loginctl unlock-sessions
#
# NOTE:
# Root privilege is required.
#
# Reference:
# * KDE bugs: #360489
#   https://bugs.kde.org/show_bug.cgi?id=360489
#
# Aaron LI
# 2017-02-07
#

for session in $(ck-list-sessions | grep '^Session' | tr -d ':'); do
    echo "* Unlock ConsoleKit session: ${session} ..."
    sudo dbus-send --system --print-reply \
        --dest="org.freedesktop.ConsoleKit" \
        /org/freedesktop/ConsoleKit/Session2 \
        org.freedesktop.ConsoleKit.Session.Unlock
done

echo "Done! Unlocked all ConsoleKit sessions!"
