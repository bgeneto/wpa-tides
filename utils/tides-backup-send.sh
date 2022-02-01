#!/bin/bash
# crontab usage: */20 * * * * /home/wpa/bin/tides-backup-send.sh > /dev/null 2>&1

# ====== BEG: custom variables ======
work_dir='/home/wpa/bin'
compress_dir='/home/wpa/bin/data'
passwd=''
url='https://<address>:<port>/s/<shareid>'
fn="tides-data-$HOSTNAME.tgz"
# ====== END: custom variables ======

# change to working folder
cd $work_dir

# compress data folder
/bin/tar -czf "$work_dir/$fn" -C "$compress_dir" .

# send compressed file to cloud
if [ -z "$passwd" ]
then
    ./cloudsend.sh -q "$work_dir/$fn" "$url"
else
    ./cloudsend.sh -q -p "$passwd" "$work_dir/$fn" "$url"
fi
