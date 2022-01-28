#!/bin/sh

sudo mst stop
sudo mst start 

sudo mlxlink -d /dev/mst/mt4121_pciconf0.1 --fec RS --fec_speed 100G
sudo mlxlink -d /dev/mst/mt4121_pciconf0.1 -a tg


sudo mlxlink -d /dev/mst/mt4121_pciconf1.1 --fec RS --fec_speed 100G
sudo mlxlink -d /dev/mst/mt4121_pciconf1.1 -a tg