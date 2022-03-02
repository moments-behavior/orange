#!/bin/sh

#sudo mst stop
sudo mst start 

sudo mlxlink -d /dev/mst/mt4119_pciconf0 --fec RS --fec_speed 25G
sudo mlxlink -d /dev/mst/mt4119_pciconf0 -a tg


sudo mlxlink -d /dev/mst/mt4119_pciconf0.1 --fec RS --fec_speed 25G
sudo mlxlink -d /dev/mst/mt4119_pciconf0.1 -a tg