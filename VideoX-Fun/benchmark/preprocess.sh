#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p ./start_frames

# 遍历 ./videos 目录下的所有视频文件
for video in ./videos/*; do
    if [[ -f "$video" ]]; then
        # 提取文件名（不含路径和扩展名）
        filename=$(basename "$video")
        name="${filename%.*}"

        # 提取第一帧并保存
        ffmpeg -y -i "$video" -vf "select=eq(n\,0)" -q:v 2 "./start_frames/${name}.jpg"
    fi
done
