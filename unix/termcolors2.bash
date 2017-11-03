#!/usr/bin/env bash
#
# Credits:
# [1] http://frexx.de/xterm-256-notes/data/colortable16.sh
# [2] https://wiki.archlinux.org/index.php/X_resources
#
# Modified by Aaron Griffin, and Kazuo Teramoto
#


FGNAMES=(' black ' '  red  ' ' green ' ' yellow' '  blue ' 'magenta' '  cyan ' ' white ')
BGNAMES=('DFT' 'BLK' 'RED' 'GRN' 'YEL' 'BLU' 'MAG' 'CYN' 'WHT')
#echo "     ----------------------------------------------------------------------------"
echo "     ┌──────────────────────────────────────────────────────────────────────────┐"
for b in $(seq 0 8); do
    if [ "$b" -gt 0 ]; then
        bg=$(($b+39))
    fi

    #echo -en "\033[0m ${BGNAMES[$b]} : "
    echo -en "\033[0m ${BGNAMES[$b]} │ "

    for f in $(seq 0 7); do
        fg=$(($f+30))
        echo -en "\033[${bg}m\033[${fg}m ${FGNAMES[$f]} "
    done

    #echo -en "\033[0m :"
    #echo -en "\033[0m\n\033[0m     : "
    echo -en "\033[0m │"
    echo -en "\033[0m\n\033[0m     │ "

    for f in $(seq 0 7); do
        fg=$(($f+30))
        echo -en "\033[${bg}m\033[1;${fg}m ${FGNAMES[$f]} "
    done

    #echo -en "\033[0m :"
    echo -en "\033[0m │"
    echo -e "\033[0m"

  if [ "$b" -lt 8 ]; then
    #echo "     ----------------------------------------------------------------------------"
    echo "     ├──────────────────────────────────────────────────────────────────────────┤"
  fi
done
#echo "     ----------------------------------------------------------------------------"
echo "     └──────────────────────────────────────────────────────────────────────────┘"
