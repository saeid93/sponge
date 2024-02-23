#!/bin/bash

function load_tester_key() {
    PUBLIC_KEY_FILE=~/infrastructure/hack/load-tester-key
    SSH_DIR=~/.ssh
    AUTHORIZED_KEYS_FILE=$SSH_DIR/authorized_keys

    mkdir -p $SSH_DIR
    chmod 700 $SSH_DIR

    cat $PUBLIC_KEY_FILE >> $AUTHORIZED_KEYS_FILE
    chmod 600 $AUTHORIZED_KEYS_FILE
    echo "Public key added to authorized hosts on the local machine."
}

load_tester_key