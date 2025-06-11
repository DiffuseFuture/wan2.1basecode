from cgi import test
import json
import os


f = open("./benchmark/test.json","r")
data = f.read()
test_data = json.loads(data)

for item in test_data:
    file_path = item["file_path"]
    vname = file_path.split("/")[-1]
    prompt = item["text"]

    command = "torchrun --nproc-per-node=2 examples/wan2.1_fun/predict_i2v_benchmark.py --ref \"/workspace/VideoX-Fun/benchmark/start_frames/{}\" --prompt \"{}\" ".format(vname[:-4]+".jpg",str(prompt))
    # command = "torchrun --nproc-per-node=2 examples/wan2.1_fun/predict_i2v.py"

    print(command)
    os.system(command)

