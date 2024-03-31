#!/bin/bash

function upload_all() {
    gsutil cp -rn ~/sponge/data/results gs://sponge/
}

upload_all
