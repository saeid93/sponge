#!/bin/bash

# syncing time solution two
function sync_time_2() {
    sudo ufw allow 123/tcp
    sudo ufw allow 123/utp
    sudo apt update
    sudo apt install ntp
    sudo systemctl restart ntp
    echo "Syncing time"
    echo
}

sync_time_2