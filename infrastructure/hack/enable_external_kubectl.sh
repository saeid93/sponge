#!/bin/bash

function enable_external() {
    # enabling external kubectl access
    file_path="/var/snap/microk8s/current/certs/csr.conf.template"
    new_line="IP.100 = $PUBLIC_IP"
    line_numbers=($(grep -n "IP.2" "$file_path" | cut -d':' -f1))
    last_line_number="${line_numbers[-1]}"
    echo "$new_line" | sudo sed -i "$last_line_number a $(cat)" "$file_path"
    echo "Line \"$new_line\" added after last occurrence of IP.2 in $file_path"
}

enable_external
