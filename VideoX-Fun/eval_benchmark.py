import torch
from utils.calculate_fvd import calculate_fvd
from utils.calculate_psnr import calculate_psnr
from utils.calculate_ssim import calculate_ssim
from utils.calculate_lpips import calculate_lpips

from torch.utils.data import DataLoader
from utils.util import VideoPairDataset 
device = torch.device("cuda")
only_final = True

import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="示例脚本参数解析")

    # 添加参数示例
    parser.add_argument(
        "--root_predict", 
        type=str, 
        default="./samples/wan-videos-fun-i2v/",
    )
    parser.add_argument(
        "--root_benchmark", 
        type=str, 
        default="./benchmark/videos/",
    )
    return parser.parse_args()

args = parse_args()
# Example usage
if __name__ == "__main__":
    dataset = VideoPairDataset(root_benchmark=args.root_benchmark,root_predict=args.root_predict, sequence_len=15)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    results = []
    for videos1, videos2 in dataloader:
        result = {}

        # inputs and targets are [batch_size, sequence_len, 64, 64, 3]
        print(videos1.shape, videos2.shape)

        result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
        result['ssim'] = calculate_ssim(videos1, videos2, only_final=only_final)
        result['psnr'] = calculate_psnr(videos1, videos2, only_final=only_final)
        result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=only_final)
        results.append(result)
    
    # 每段视频的metric会被保存在result.json中
    with open("eval_result.json","w") as f:
        f.write(json.dumps(results, indent=4))