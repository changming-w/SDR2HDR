#!/bin/bash
# 批量处理 SDR 视频转 HDR

INPUT_DIR="/mnt/data/wcm/data/hdr/test/SDR"
OUTPUT_BASE="/mnt/data/wcm/data/hdr/test/output/hdrnet_bt709v2"
CONFIG="/mnt/data/wcm/HDRNet/options/train/Custom/train_hdrnet_bt709_sdr2hdr_small_v2.yml"
CHECKPOINT="/mnt/data/wcm/HDRNet/experiments/train_hdrnet_bt709_sdr2hdr_small_v2/models/net_g_200000.pth"
METADATA="/mnt/data/wcm/data/hdr/test/SDR/hdr10_metadata_default.json"

mkdir -p "$OUTPUT_BASE"

cd /mnt/data/wcm/HDRNet

for video in "$INPUT_DIR"/*.mp4 "$INPUT_DIR"/*.mkv "$INPUT_DIR"/*.avi "$INPUT_DIR"/*.mov; do
    [ -f "$video" ] || continue
    
    filename=$(basename "$video")
    name="${filename%.*}"
    output="$OUTPUT_BASE/${name}_hdr10.mp4"
    
    echo "=========================================="
    echo "处理: $filename"
    echo "输出: $output"
    echo "=========================================="
    
    PYTHONPATH=/mnt/data/wcm/HDRNet:$PYTHONPATH python3 scripts/infer_sdr2hdr.py \
        --input "$video" \
        --output_video "$output" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --metadata "$METADATA" \
        --fp16 --batch_size 1
    
    if [ $? -eq 0 ]; then
        echo "✓ 完成: $output"
    else
        echo "✗ 失败: $filename"
    fi
    
    echo ""
done

echo "全部处理完成！"
