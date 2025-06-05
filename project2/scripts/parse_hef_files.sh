#!/bin/bash

echo "🚀 [HEF 파일 파싱 시작]"

HEF_DIR="./compiled_model/hef"
mkdir -p "$HEF_DIR"

hef_files=("$HEF_DIR"/*.hef)
if [ ! -e "${hef_files[0]}" ]; then
    echo "⚠️  $HEF_DIR 디렉토리에 HEF 파일이 없습니다."
    exit 0
fi

success_count=0
fail_count=0
unsupported_count=0

for hef_file in "$HEF_DIR"/*.hef; do
    filename=$(basename "$hef_file")
    echo "⏳ Parsing $filename..."
    output=$(hailortcli parse-hef "$hef_file" 2>&1)
    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ Successfully parsed $filename"
        echo "----- [파싱 결과] -----"
        echo "$output"
        ((success_count++))
    else
        echo "❌ Failed to parse $filename"
        if echo "$output" | grep -q "Unsupported hef version"; then
            echo "   ⚠️ Unsupported HEF version (SDK 버전 불일치)"
            ((unsupported_count++))
        else
            echo "   Error: $(echo "$output" | grep '\[error\]' | head -1)"
        fi
        echo "----- [에러/출력 내용] -----"
        echo "$output"
        ((fail_count++))
    fi
    echo "-----------------------------------"
done

echo "🎉 파싱 완료!"
echo "✅ 성공: $success_count"
echo "❌ 실패: $fail_count"
if [ $unsupported_count -gt 0 ]; then
    echo "⚠️  SDK 버전 불일치(Unsupported HEF): $unsupported_count"
fi