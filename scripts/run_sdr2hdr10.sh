#!/usr/bin/env bash
# =============================================================================
# SDR -> HDR10 批量脚本：推理 + 音轨自动拷回（infer_sdr2hdr.py 默认行为）+ HDR10 校验
# 封装 infer_sdr2hdr.py 的「验证过的」正确调用方式（含 PYTHONPATH / 绝对 python /
# 实验配置匹配 / 音轨保留）。
#
# 默认批处理 /mnt/data/wcm/data/hdr/test/SDR 下所有 .mp4，
# 输出到 /mnt/data/wcm/data/hdr/test/output/hdr10/（同名）。
# 已转换过的文件会被 infer_sdr2hdr.py 自动跳过（断点续传状态判断）。
#
# 用法:
#   scripts/run_sdr2hdr10.sh              # 批量处理整个目录
#   scripts/run_sdr2hdr10.sh in.mp4 out.mp4   # 也可指定单个视频
# 可用环境变量覆盖: CKPT= CONFIG= CRF= PRESET= BATCH=
# 例:
#   CRF=10 scripts/run_sdr2hdr10.sh
# =============================================================================
set -euo pipefail

REPO=/mnt/data/wcm/HDRNet
PY=/home/hp/miniforge3/envs/hdrnet/bin/python

# ---- 模型（v3 paper color_v2 / 195000 iter，架构 HDRnetBT709Paper）----
CONFIG=${CONFIG:-$REPO/experiments/train_hdrnet_bt709_sdr2hdr_small_v3_paper_color_v2/train_hdrnet_bt709_sdr2hdr_small_v3_paper_color_v2.yml}
CKPT=${CKPT:-$REPO/experiments/train_hdrnet_bt709_sdr2hdr_small_v3_paper_color_v2/models/net_g_195000.pth}

# ---- 输入目录 / 输出目录 / 元数据 ----
SDR_DIR=${SDR_DIR:-/mnt/data/wcm/data/hdr/test/SDR}
OUT_DIR=${OUT_DIR:-/mnt/data/wcm/data/hdr/test/output/train_hdrnet_bt709_sdr2hdr_small_v3_paper_color_v2}
META=${META:-$SDR_DIR/hdr10_metadata_default.json}

# ---- 质量 / 性能 ----
CRF=${CRF:-12}          # 10bit HDR: 12 已肉眼无损；要更大余量可设 10
PRESET=${PRESET:-slow}
BATCH=${BATCH:-1}       # 4K 下 batch4 显存峰值约 17.8GB

mkdir -p "$OUT_DIR"

# 单视频处理函数：推理（音轨由 infer 自动拷回）+ HDR10 校验
process_one() {
  local INPUT="$1" OUTPUT="$2"
  mkdir -p "$(dirname "$OUTPUT")"

  echo "========== [1/2] SDR->HDR10 推理: $(basename "$INPUT") $(date +%H:%M:%S) =========="
  PYTHONPATH=$REPO "$PY" "$REPO/scripts/infer_sdr2hdr.py" \
    --input      "$INPUT" \
    --output_video "$OUTPUT" \
    --config     "$CONFIG" \
    --checkpoint "$CKPT" \
    --metadata   "$META" \
    --fp16 --batch_size "$BATCH" --hwaccel cuda --crf "$CRF" --preset "$PRESET"

  echo "========== [2/2] 校验 =========="
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,profile,pix_fmt,color_transfer,color_primaries,color_space,color_range \
    -of default=noprint_wrappers=1 "$OUTPUT"
  echo -n "HDR10 SEI: "
  ffprobe -v error -select_streams v:0 -read_intervals "%+#1" -show_frames \
    -show_entries frame=side_data_list -of json "$OUTPUT" 2>/dev/null \
    | grep -oiE "Mastering display metadata|Content light level metadata" | sort -u | paste -sd' + '

  echo "========== 完成: $OUTPUT =========="
}

FAILED=()

if [ $# -ge 1 ]; then
  # ---- 单视频模式（兼容旧用法）----
  INPUT="$1"
  OUTPUT=${2:-$OUT_DIR/$(basename "${INPUT%.*}").mp4}
  process_one "$INPUT" "$OUTPUT"
else
  # ---- 批量模式：遍历输入目录所有 .mp4 ----
  TOTAL=$(find "$SDR_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
  echo "========== 批量处理 $SDR_DIR 下 $TOTAL 个视频 -> $OUT_DIR =========="
  IDX=0
  for INPUT in "$SDR_DIR"/*.mp4; do
    IDX=$((IDX + 1))
    OUTPUT="$OUT_DIR/$(basename "${INPUT%.*}").mp4"
    echo ""
    echo "########## [$IDX/$TOTAL] ##########"
    if process_one "$INPUT" "$OUTPUT"; then
      echo "[OK] $OUTPUT"
    else
      echo "[FAIL] $INPUT"
      FAILED+=("$(basename "$INPUT")")
    fi
  done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "========== 有 ${#FAILED[@]} 个视频失败: ${FAILED[*]} =========="
  exit 1
fi
echo ""
echo "========== 全部完成 =========="
