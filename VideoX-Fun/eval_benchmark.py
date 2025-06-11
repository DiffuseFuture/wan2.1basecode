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
result = {}


# Example usage
if __name__ == "__main__":
    dataset = VideoPairDataset(root_predict="./examples-2/pred", sequence_len=15)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for videos1, videos2 in dataloader:
        # inputs and targets are [batch_size, sequence_len, 64, 64, 3]
        print(videos1.shape, videos2.shape)

        result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
        result['ssim'] = calculate_ssim(videos1, videos2, only_final=only_final)
        result['psnr'] = calculate_psnr(videos1, videos2, only_final=only_final)
        result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=only_final)

        print(json.dumps(result, indent=4))
        break
