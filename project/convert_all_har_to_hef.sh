#!/bin/bash

HAR_DIR="./compiled_model"
HEF_DIR="./compiled_model"

for har_path in "$HAR_DIR"/lstm_*.har; do
    base=$(basename "$har_path" .har)
    hef_path="$HEF_DIR/${base}.hef"

    echo "➡️ Converting $har_path to $hef_path ..."
    hailo optimize "$har_path" \
        --output "$hef_path" \
        --hw-arch hailo8 \
        --use-random-calib-set
    echo "✅ Done: $hef_path"
    echo ""
done
