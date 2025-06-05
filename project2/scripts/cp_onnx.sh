#!/bin/bash

SOURCE_BASE_DIR="./onnx_models"
DEST_DIR="/local/shared_with_docker"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Check if the source directory exists
if [ ! -d "$SOURCE_BASE_DIR" ]; then
    echo "Source directory $SOURCE_BASE_DIR not found."
    exit 1
fi

# Iterate over each subdirectory in SOURCE_BASE_DIR
for SUBDIR_PATH in "$SOURCE_BASE_DIR"/*/; do
    # Check if it's a directory
    if [ -d "$SUBDIR_PATH" ]; then
        SUBDIR_NAME=$(basename "$SUBDIR_PATH")

        # Iterate over .onnx files in the current subdirectory
        for ONNX_FILE_PATH in "$SUBDIR_PATH"*.onnx; do
            # Check if the .onnx file exists (glob returns pattern if no match)
            if [ -f "$ONNX_FILE_PATH" ]; then
                ONNX_FILENAME=$(basename "$ONNX_FILE_PATH")
                DEST_FILE_PATH="$DEST_DIR/${SUBDIR_NAME}_${ONNX_FILENAME}"
                echo "cp \"$ONNX_FILE_PATH\" \"$DEST_FILE_PATH\""
                cp "$ONNX_FILE_PATH" "$DEST_FILE_PATH"

                DEST_FILE_PATH="$SOURCE_BASE_DIR/${SUBDIR_NAME}_${ONNX_FILENAME}"
                echo "cp \"$ONNX_FILE_PATH\" \"$DEST_FILE_PATH\""
                cp "$ONNX_FILE_PATH" "$DEST_FILE_PATH"
            fi
        done
    fi
done

echo "ONNX file copying complete."