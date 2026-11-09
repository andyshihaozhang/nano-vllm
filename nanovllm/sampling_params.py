from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0    # token采样温度，用于控制随机性
    max_tokens: int = 64        # 最大生成token长度
    ignore_eos: bool = False    # 是否忽略生成eos标记

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted" # 不允许随机采样
