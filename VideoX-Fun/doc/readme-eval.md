# 用于生成视频质量评估(目前支持FID，PSNR，LPIPS，FVD)

![Alt text](../asset/metrics.png)

# 使用方法

+ step 1: 推理生成视频


    python infer_benchmark.py


+ step 2: 对生成结果进行评估


    python eval_benchmark.py

生成结果会被保存在eval_result.json文件中。
