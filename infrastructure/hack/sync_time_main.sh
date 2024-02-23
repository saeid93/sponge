#!/bin/bash

function sync_time_load_main () {
  sudo apt-get update
  sudo apt-get install -y ntp
  sudo systemctl restart ntp
  echo "NTP Servers and Synchronization Status:"
}

sync_time_load_main
