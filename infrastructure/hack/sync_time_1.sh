#!/bin/bash

# syncing time solution one
function sync_time_1() {
    sudo apt install systemd-timesyncd
    sudo timedatectl set-ntp 1
    sudo systemctl restart systemd-timedated.service
    echo "Syncing time"
    echo
}

sync_time_1