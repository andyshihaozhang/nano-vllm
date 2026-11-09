# 模型加载-加载下载的模型权重
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

'''
模型权重加载函数
'''
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys(): # 文件权重路径（通过文件路径获取） model.layers.0.mlp.gate_proj.weight
                for k in packed_modules_mapping: # "gate_proj": ("gate_up_proj", 0),
                    if k in weight_name: # gate_proj
                        v, shard_id = packed_modules_mapping[k] # gate_up_proj 0 gate_proj
                        param_name = weight_name.replace(k, v) # 内存权重路径（通过对象调用获取） model.layers.0.mlp.gate_up_proj.weight
                        param = model.get_parameter(param_name) # 内存权重对象
                        weight_loader = getattr(param, "weight_loader") # 获取内存权重对象对应权重加载函数（已在LinearBase初始化时绑定）
                        weight_loader(param, f.get_tensor(weight_name), shard_id) # 文件权重 -> 内存权重 
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
