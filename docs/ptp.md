## Configure PTP 

### Local PTP 

0. install Linux PTP package

```
sudo apt install linuxptp
```

1. set boundary clock 
If you don't have to `ptp4l.conf` file, create it first `/etc/ptp4l.conf`, and edit the file

```
[global]
verbose 1
boundary_clock_jbod 1
```

2. Synchronize NICs

If you only using one NIC card, then

```
sudo ptp4l -i enp66s0f0np0 -i enp66s0f1np1 -f /etc/ptp4l.conf
```
This will sync the ports of the NIC. `enp66s0f0np0, enp66s0f1np1` are the names of the port for instance, you will need to change them to your port name. 

If you need to sync multiple NICs, you not only need to sync all the ports on the NIC, but across NICs, for intance,  

```
sudo ptp4l -i enp66s0f0np0 -i enp66s0f1np1 -i enp97s0f0np0 -i enp97s0f1np1 -f /etc/ptp4l.conf
```
in another termial, run
```
sudo phc2sys -a -rr -m
```
this will sync the computer time and consider it as a time source


### Network PTP using switch 
If you are using network switch, please enable PTP on all the ports connected to either the camera or host NICs. Here is an example of setting it with Arista and Mellanox (Onyx) switch.

Once logged in: 

```
en
```

```
conf t
```

```
show ptp 
```

if ptp is disabled: 

```
ptp mode boundary 
```

config each port 

```
show interface
``` 


Example to enable ptp on port 5
```
interface Ethernet 5
ptp enable 

```

To config multiple ports:

```
interface Ethernet49/1-Ethernet60/1  
ptp enable 
```

```
interface Ethernet1-Ethernet48  
ptp enable 
```

```
show int sta
```

if there is errdisabled, try to restart the port

```
interface Ethernet ##
shutdown 
no shutdown 
```

To disable the spanning tree 

```
spanning-tree bpduguard disable
```


## Mellanox

[Document](https://safe.nrao.edu/wiki/pub/Beamformer/DDLTestingCommModeOne/MLNX-OS_SW_Eth_3.2.0506_Command_Reference_Guide.pdf)

Userful command: 

```
show interface ethernet 1/11
```

```
enable 
configure terminal 
interface ethernet 1/11
shutdown
mtu 9216
speed 25000
fec rs-fec
no shutdown
```

To config multiple ports: 
```
interface ethernet 1/1-1/16 shutdown  
interface ethernet 1/1-1/16 mtu 9216  
interface ethernet 1/1-1/16 fec rs-fec  
interface ethernet 1/1-1/16 speed 25G  
interface ethernet 1/1-1/16 no shutdown

```


