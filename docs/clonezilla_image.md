## Restore disk from Clonezilla image

An Ubuntu image is available with preinstalled `orange` and the labeling app `red`. Please contact [Jinyao Yan](yanj11@janelia.hhmi.org) if you wish to use it. Be aware that you will lose all the data orginally on the disk. Backup data before following these instructions, 

1. Download the image zip file, unzip it onto an Ubuntu local disk or USB.
2. Make a [Clonezilla live USB](https://clonezilla.org/clonezilla-live.php).
3. [Restore disk image](https://clonezilla.org//fine-print-live-doc.php?path=clonezilla-live/doc/02_Restore_disk_image)

After successfully restoring the image, in BIOS, enable **PCIE Above 4G Decoding** and **Resizable Bar**. You can try upgrade the BIOS if the current BIOS doesn't support them.

Here is the system stack of this image,
```
ubuntu_version: 22.04.4
kernel_version: 6.5.0-44-generic
nvidia_driver_version: 535.183.06
esdk_version: 2.55.02
cuda_version: 12.2
ffmpeg_version: 4.4
opencv_version: 4.10
tensorrt_version: 10.0.1.6
```
