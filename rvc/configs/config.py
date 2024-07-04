import mlx.core as mx
import json
import os

version_config_list = [
    "v1/32000.json",
    "v1/40000.json",
    "v1/48000.json",
    "v2/48000.json",
    "v2/32000.json",
]

def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper

@singleton_variable
class Config:
    def __init__(self):
        self.device = "gpu"  # MLX automatically uses the available GPU
        self.is_half = True
        self.n_cpu = 0
        self.json_config = self.load_config_json()
        self.instead = ""
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            with open(f"rvc/configs/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    def use_fp32_config(self):
        print("Using FP32 config instead of FP16")
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
            with open(f"rvc/configs/{config_file}", "r") as f:
                strr = f.read().replace("true", "false")
            with open(f"rvc/configs/{config_file}", "w") as f:
                f.write(strr)
        with open("rvc/train/preprocess/preprocess.py", "r") as f:
            strr = f.read().replace("3.7", "3.0")
        with open("rvc/train/preprocess/preprocess.py", "w") as f:
            f.write(strr)

    def device_config(self) -> tuple:
        # MLX automatically uses the GPU if available, so we don't need to check
        self.is_half = True  # MLX uses float16 by default on GPU
        
        if self.n_cpu == 0:
            self.n_cpu = os.cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        return x_pad, x_query, x_center, x_max

# Note: The following functions are not directly applicable to MLX
# as it doesn't provide GPU memory information in the same way as PyTorch.
# You may need to implement alternative methods or remove these if not needed.

def max_vram_gpu(gpu):
    return "N/A"  # MLX doesn't provide this information

def get_gpu_info():
    return "MLX automatically uses the available GPU. Detailed GPU information is not provided."
