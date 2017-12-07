#!/bin/sh
#
# fbsd-install-iso2img.sh
# Original version by Dario Freni 9/2006
# Enhancements by Clifton Royston 3/2009.
# License: Beerware
#
# Credit: https://lists.freebsd.org/pipermail/freebsd-stable/2009-March/048762.html
#

# You can set some variables here. Edit them to fit your needs.

# Set serial variable to 0 if you don't want serial console at all,
# 1 if you want comconsole and 2 if you want comconsole and vidconsole
serial=0

# Set nofstab=1 to not create any initial fstab on the USB drive;
# this makes the next two settings largely irrelevant.
nofstab=0

# Set rootperm=rw for root fs to mount r/w from the USB drive
#  (Should be unnecessary.)
rootperm=ro

# Set USBLABEL here or with -L label to label the image file system, 
#  to help the loader find the root file system when booting; 
#  otherwise the USB must come up as da0 to finish loading successfully.
USBLABEL=
lbparams=

# Set dopause=1 here or with -p to pause and allow review or editing of
#  the flash image before finalizing the image.
dopause=0

pause() {
 echo "Press enter to continue"
 read foo
}


set -u

if [ $# -ge 3 ]; then
  flag=$1
  if [ ${flag} = "-p" ]; then
    dopause=1
    shift
    flag=$1
  fi
  if [ ${flag} = "-n" ]; then
    nofstab=1
    shift
    flag=$1
  fi
  if [ ${flag} = "-L" ]; then
    shift;
    USBLABEL=$1; shift
    lbparams="-L ${USBLABEL}"
  fi
fi
if [ $# -lt 2 ]; then
  echo "Usage: $0 [-p] [-n] [-L vollabel] source-iso-path output-img-path"
  echo "	-p	pause for review before finalizing image"
  echo "	-n	don't update the /etc/fstab within the image"
  echo "	-L	set file system label on image, to help loader find it"
  exit 1
fi

isoimage=$1; shift
imgoutfile=$1; shift

export tmpdir=$(mktemp -d -t fbsdmount)

# Temp file and directory to be used later
export tmpfile=$(mktemp -t bsdmount)

export isodev=$(mdconfig -a -t vnode -f ${isoimage})

echo "#### Building bootable UFS image ####"

ISOSIZE=$(du -k ${isoimage} | awk '{print $1}')
SECTS=$((($ISOSIZE + ($ISOSIZE/5))*2))

# Root partition size

echo "Initializing image..."
dd if=/dev/zero of=${imgoutfile} count=${SECTS}
ls -l ${imgoutfile}
export imgdev=$(mdconfig -a -t vnode -f ${imgoutfile})

bsdlabel -w -B ${imgdev}
newfs -O1 ${lbparams} /dev/${imgdev}a

mkdir -p ${tmpdir}/iso ${tmpdir}/img

mount -r -t cd9660 /dev/${isodev} ${tmpdir}/iso
mount /dev/${imgdev}a ${tmpdir}/img

echo "Copying files to the image via cpio"
( cd ${tmpdir}/iso && find . -print -depth | cpio -dump ${tmpdir}/img )
# Dump doesn't work from an ISO file system, too bad.
# echo "Copying files to the image via dump/restore..."
## dump -0f - /dev/${isodev} | (cd ${tmpdir}/img && restore -r -f - ) 

#bzcat ${tmpdir}/iso/dist/root.dist.bz2 | mtree -PUr -p ${tmpdir}/img 2>&1 > /dev/null

if [ ${nofstab} -ne 1 ]; then
  echo "Saving original /etc/fstab as /etc/fstab.orig"
  mv ${tmpdir}/img/etc/fstab  ${tmpdir}/img/etc/fstab.orig
  echo "Replacing /etc/fstab, so loader can find root filesystem on flash!"
  if [ "${USBLABEL}" != "" ]; then
    echo "/dev/ufs/${USBLABEL} / ufs ${rootperm} 0 0" > ${tmpdir}/img/etc/fstab
    ## echo "devfs /dev devfs rw 0 0" >> ${tmpdir}/img/etc/fstab
  else 
    echo "/dev/da0a / ufs ${rootperm} 0 0" > ${tmpdir}/img/etc/fstab
    ## echo "devfs /dev devfs rw 0 0" >> ${tmpdir}/img/etc/fstab
  fi
else
  echo "Skipping write of image /etc/fstab"

fi

if [ ${serial} -eq 2 ]; then
        mv ${tmpdir}/img/boot.config ${tmpdir}/img/boot.config.orig
        mv ${tmpdir}/img/boot/loader.conf ${tmpdir}/img/boot/loader.conf.orig
        echo "-D" > ${tmpdir}/img/boot.config
        echo 'console="comconsole, vidconsole"' >> ${tmpdir}/img/boot/loader.conf
elif [ ${serial} -eq 1 ]; then
        mv ${tmpdir}/img/boot.config ${tmpdir}/img/boot.config.orig
        mv ${tmpdir}/img/boot/loader.conf ${tmpdir}/img/boot/loader.conf.orig
        echo "-h" > ${tmpdir}/img/boot.config
        echo 'console="comconsole"' >> ${tmpdir}/img/boot/loader.conf
fi

if [ ${dopause} -eq 1 ]; then
  echo "Pausing to allow manual review and modification of image file:"
  echo "Image is located in ${tmpdir}/img"
  echo "If you need to fix up ${tmpdir}/img/etc/fstab, now is the time."
  pause
fi


cleanup() {
    umount ${tmpdir}/iso
    mdconfig -d -u ${isodev}
    umount ${tmpdir}/img
    mdconfig -d -u ${imgdev}
    rm -rf ${tmpdir} ${tmpfile}
}

cleanup

ls -lh ${imgoutfile}

echo "To write the image to flash, use dd, for example:"
echo "   dd if=${imgoutfile} of=/dev/da0 bs=4M"
