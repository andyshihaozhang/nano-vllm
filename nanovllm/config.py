# 推理引擎配置
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str                              # 本地模型目录
    max_num_batched_tokens: int = 16384     # 单批次（所有请求）最大可接受token数（Token Budget）
    max_num_seqs: int = 512                 # 最大请求数
    max_model_len: int = 4096               # 单请求最大token数【包含输入+输出token数量】
    gpu_memory_utilization: float = 0.9     # 显示利用率，估算显存可用空间大小【KV Cache 预算 ≈ 显存总量 × 利用率 − 当前已用显存 −（预热峰值占用 − 当前张量占用）】
    tensor_parallel_size: int = 1           # 张量并行规模，通常对应 GPU 和工作进程数量【模型的部分权重和计算被拆到n张 GPU，由n个工作进程协作完成一次前向计算】
    enforce_eager: bool = False             # 启用 CUDA Graph 开关
    hf_config: AutoConfig | None = None     # 从模型目录读取的 Hugging Face 模型结构配置
    eos: int = -1                           # 结束 token 的 ID，初始化时由 tokenizer 填入
    kvcache_block_size: int = 256           # 每个KV block容纳的token数
    num_kvcache_blocks: int = -1            # KV Cache 物理块总数，由显存估算结果填入

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0   # flash_attn_with_kvcache 文档写明，传入 block_table（分页 KV）时 page_block_size 必须是 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8  # TP上限，即单机8卡的上限
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
