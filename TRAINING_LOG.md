# HDRNet 训练记录

## 训练任务

| ID | 配置文件 | 开始日期 | 状态 | 数据集 | 说明 |
|----|--------|--------|------|-------|------|
| v2 | train_hdrnet_bt709_sdr2hdr_small_v2.yml | - | 已完成 | 4K (hdrframes + dark_scene_1200) | 原始 4K 版本，色度矩阵改为 BT.709 |
| v3_1080p | train_hdrnet_bt709_sdr2hdr_small_v3_1080p.yml | 2026-08-19 | 待训练 | 1080p (light_scene_1080p + dark_scene_1200_1080p) | 降采样加快训练，以 v2 205k 权重预训练 |
