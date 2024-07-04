import os
import mlx.core as mx
from collections import OrderedDict

def extract(ckpt):
    a = ckpt["model"]
    opt = OrderedDict()
    opt["weight"] = {}
    for key in a.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = a[key]
    return opt

def model_blender(name, path1, path2, ratio):
    try:
        message = f"Model {path1} and {path2} are merged with alpha {ratio}."
        ckpt1 = mx.load(path1)
        ckpt2 = mx.load(path2)
        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]

        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."

        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = mx.cast(
                    ratio * mx.array(ckpt1[key][:min_shape0], dtype=mx.float32) +
                    (1 - ratio) * mx.array(ckpt2[key][:min_shape0], dtype=mx.float32),
                    mx.float16
                )
            else:
                opt["weight"][key] = mx.cast(
                    ratio * mx.array(ckpt1[key], dtype=mx.float32) +
                    (1 - ratio) * mx.array(ckpt2[key], dtype=mx.float32),
                    mx.float16
                )

        opt["config"] = cfg
        opt["sr"] = message
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["info"] = message

        mx.save(os.path.join("logs", f"{name}.npz"), opt)
        print(message)
        return message, os.path.join("logs", f"{name}.npz")
    except Exception as error:
        print(error)
        return error