import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import evaluation.lpips as lpips

# ignore torchvision UserWarning of 'weights'
import warnings
warnings.filterwarnings("ignore")

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # value range [0, 1] -> [-1, 1]
    x = x * 2 - 1

    return x

def calculate_lpips(videos1, videos2, device, only_final=False):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    print("calculate_lpips...")

    assert videos1.shape == videos2.shape

    # videos [batch_size, timestamps, channel, h, w]

    # support grayscale input, if grayscale -> channel*3
    # value range [0, 1] -> [-1, 1]
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    lpips_results = []

    loss_fn.to(device)

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)

            # calculate lpips of a video
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    
    lpips_results = np.array(lpips_results)
    
    lpips = []
    lpips_std = []

    if only_final:

        lpips.append(np.mean(lpips_results))
        lpips_std.append(np.std(lpips_results))

    else:

        for clip_timestamp in range(len(video1)):
            lpips.append(np.mean(lpips_results[:,clip_timestamp]))
            lpips_std.append(np.std(lpips_results[:,clip_timestamp]))

    result = {
        "value": lpips,
        "value_std": lpips_std,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    result = calculate_lpips(videos1, videos2, device)
    print("[lpips avg]", result["value"])
    print("[lpips std]", result["value_std"])

if __name__ == "__main__":
    main()