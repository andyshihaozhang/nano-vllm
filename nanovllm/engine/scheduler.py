# 通过队列维护待执行的请求，组织并下发每一步需要执行的请求

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    '''
    推理调度预处理:
        调度当前轮处理的token（num_scheduled_tokens）
    - scheduled_seqs: 
    - is_prefill: 当前流程是否为 prefill 流程
    '''
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill（支持分块prefill-chunked prefill）
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # 批处理token数上限判定
            remaining = self.max_num_batched_tokens - num_batched_tokens  # 剩余待处理token数 = 最大批处理token数 - 当前已经处理token数
            if remaining == 0:
                break
            # KVCache分配判定
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq) # 前缀缓存token
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size # 新token = seq所有token - 前缀缓存token
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens # 新token = seq所有token - seq已完成的token
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            # 分配KVCache缓存空间
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq): # 缓存空间不足
                if self.running:
                    self.preempt(self.running.pop()) # 抢占队尾缓存空间
                else:
                    self.preempt(seq) # 没有可抢占空间则牺牲自己
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    '''
    调度抢占处理:
        decode阶段生成新token会导致KV缓存扩大，缓存空间不足会导致抢占
    '''
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    '''
    推理调度尾处理:
        资源回收释放
    '''
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            # 更新 已处理token 和 当前处理的token
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 释放资源
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
