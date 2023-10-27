#!/bin/bash

function upload_all_models() {
    gsutil cp -rn /mnt/myshareddir 'gs://ipa-models/myshareddir/*'
}

upload_all_models