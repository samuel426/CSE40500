#!/bin/bash

# Default values
HAR_DIR="./compiled_model"
HEF_DIR="./compiled_model"
HW_ARCH="hailo8l"
USE_RANDOM_CALIB=true
CHECK_COMPATIBILITY=true

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Convert all HAR files to HEF format"
    echo ""
    echo "Options:"
    echo "  -i, --input DIR      Input directory containing HAR files (default: $HAR_DIR)"
    echo "  -o, --output DIR     Output directory for HEF files (default: $HEF_DIR)"
    echo "  -a, --arch ARCH      Hardware architecture (default: $HW_ARCH)"
    echo "  -r, --no-random      Disable random calibration set"
    echo "  -s, --skip-check     Skip compatibility check"
    echo "  -h, --help           Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            HAR_DIR="$2"
            shift 2
            ;;
        -o|--output)
            HEF_DIR="$2"
            shift 2
            ;;
        -a|--arch)
            HW_ARCH="$2"
            shift 2
            ;;
        -r|--no-random)
            USE_RANDOM_CALIB=false
            shift
            ;;
        -s|--skip-check)
            CHECK_COMPATIBILITY=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$HEF_DIR"

# Check if HAR files exist
har_files=("$HAR_DIR"/*.har)
if [ ! -e "${har_files[0]}" ]; then
    echo "Error: No .har files found in $HAR_DIR"
    exit 1
fi

# Check HailoRT version compatibility
check_hailo_version() {
    local hailo_version=$(hailo_get_version 2>/dev/null || echo "unknown")
    echo "ℹ️ HailoRT version: $hailo_version"
    
    # You may need to adjust this based on your specific version requirements
    if [[ "$hailo_version" == "unknown" ]]; then
        echo "⚠️ Warning: Could not determine HailoRT version."
    fi
}

if [ "$CHECK_COMPATIBILITY" = true ]; then
    check_hailo_version
fi

echo "🔄 Converting HAR files to HEF format..."
echo "📁 Input directory: $HAR_DIR"
echo "📁 Output directory: $HEF_DIR"
echo "🛠️ Hardware architecture: $HW_ARCH"
echo ""

# Counter for successful conversions
success_count=0
total_count=0
failed_files=()

for har_path in "$HAR_DIR"/*.har; do
    base=$(basename "$har_path" .har)
    hef_path="$HEF_DIR/${base}.hef"
    total_count=$((total_count + 1))

    echo "➡️ Converting $(basename "$har_path") to $(basename "$hef_path") ..."
    
    cmd="hailo optimize \"$har_path\" --output \"$hef_path\" --hw-arch $HW_ARCH"
    if [ "$USE_RANDOM_CALIB" = true ]; then
        cmd="$cmd --use-random-calib-set"
    fi
    
    # Capture both stdout and stderr
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Done: $hef_path"
        success_count=$((success_count + 1))
    else
        echo "❌ Failed to convert $har_path"
        echo "$output" | grep -i "error" | head -5  # Show first 5 error lines
        
        # Check for specific compatibility errors
        if echo "$output" | grep -q "Unsupported hef version"; then
            echo "⚠️ Version compatibility issue detected. Try:"
            echo "   1. Update HailoRT to the latest version"
            echo "   2. Use a compatible version of the Hailo compiler"
            echo "   3. Contact Hailo support for assistance"
        fi
        
        failed_files+=("$(basename "$har_path")")
    fi
    echo ""
done

echo "🎉 Conversion complete: $success_count/$total_count files successfully converted"

if [ ${#failed_files[@]} -gt 0 ]; then
    echo "❌ Failed files:"
    for file in "${failed_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "   1. Check if you're using the correct hardware architecture flag (current: $HW_ARCH)"
    echo "   2. Ensure HailoRT version is compatible with your HAR files"
    echo "   3. Try recompiling the models with a compatible version of the compiler"
    echo "   4. Use 'hailo_get_version' to check your current HailoRT version"
    exit 1
fi
