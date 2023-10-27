#!/bin/bash

function upload_node_models(){
    gsutil cp -rn /mnt/myshareddir/$SOURCE/$NODE/* 'gs://ipa-models/myshareddir/'$SOURCE/$NODE
}

upload_node_models