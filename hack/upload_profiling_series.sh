#!/bin/bash

function upload_profiling_series(){
    gsutil cp -rn ~/sponge/data/results/profiling/nodes/series/$SERIES gs://sponge/results/profiling/nodes/series/$SERIES
}

upload_profiling_series
