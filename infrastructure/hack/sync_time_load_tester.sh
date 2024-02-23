#!/bin/bash

function sync_time_load_tester () {
  sudo apt-get update
  sudo apt-get install -y ntp
  echo "server $PRIMARY_IP" | sudo tee -a /etc/ntp.conf
  sudo systemctl restart ntp
  echo "NTP Servers and Synchronization Status:"
}

sync_time_load_tester
