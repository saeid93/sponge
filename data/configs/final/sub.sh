#!/bin/bash

text_to_add=$(cat <<EOF

# ----------- whether to read models from storage or not - should be defined per node -------------
from_storage:
  - true
  - true
EOF
)

for file in *.yaml; do
    if [ -f "$file" ]; then
        sed -i '/^logs_enabled:/,$d' "$file"
        echo "$text_to_add" >> "$file"
        echo "logs_enabled: false" >> "$file"
    fi
done
