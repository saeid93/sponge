# considering that you have already connected to the object storage
gsutil cp -rn ./results/final/metaseries/$1 gs://ipa-results/results/final/metaseries/$1
