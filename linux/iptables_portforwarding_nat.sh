#!/bin/sh
#
# Port forwarding from one address to another address in the same network,
# using source and destination network address translation (SNAT & DNAT).
#
# The machine A performs this port forwarding to the target machine B,
# which is in the same network as A.
# The machine A behaves like a proxy, which allows e.g., external machine
# access the services (e.g., SSH) on machine B which only allow access
# from the internal network.
#
#
# References:
# [1] How to do the port forwarding from one ip to another ip in the same network?
#     https://serverfault.com/a/586553/387898
# [2] Source and Destination Network Address Translation with iptables
#     https://thewiringcloset.wordpress.com/2013/03/27/linux-iptable-snat-dnat/
# [3] How to List and Delete IPtables Firewall Rules
#     https://www.digitalocean.com/community/tutorials/how-to-list-and-delete-iptables-firewall-rules
#
#
# Weitian LI
# 2016-11-29
#


# Enable IP forwarding
sysctl net.ipv4.ip_forward=1

# Save current rules
iptables-save > iptables_rules.txt

# Set default chain policy
iptables -P INPUT ACCEPT
iptables -P FORWARD ACCEPT
iptables -P OUTPUT ACCEPT

# Flush existing rules
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
iptables -F
iptables -X

# Port forwarding using SNAT & DNAT
THIS_IP="192.168.1.234"
THIS_PORT="21127"
TARGET_IP="192.168.1.248"
TARGET_PORT="9999"
echo "Port forwarding: ${THIS_IP}:${THIS_PORT} <-> ${TARGET_IP}:${TARGET_PORT}"
iptables -t nat -A PREROUTING \
         -p tcp --dport ${THIS_PORT} \
         -j DNAT --to-destination ${TARGET_IP}:${TARGET_PORT}
iptables -t nat -A POSTROUTING \
         -p tcp -d ${TARGET_IP} --dport ${TARGET_PORT} \
         -j SNAT --to-source ${THIS_IP}:${THIS_PORT}
