#!/bin/bash

function upload_profiling_series(){
    gsutil cp -rn ~/ipa-private/data/results/profiling/nodes/series/$SERIES gs://ipa-results/results/profiling/nodes/series/$SERIES
}

upload_profiling_series
