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

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        token_budget = self.max_num_batched_tokens

        # running
        req_idx=0
        while self.running and req_idx<len(self.running) and len(scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = self.running[req_idx]
            assert seq.block_table
            if(seq.num_cached_tokens>=seq.num_prompt_tokens):
                # decode
                while not self.block_manager.can_append(seq):
                    if(req_idx==len(self.running)-1):
                        self.running.pop()
                        self.preempt(seq)
                        break
                    else:
                        self.preempt(self.running.pop())
                else:
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    token_budget= token_budget - seq.num_scheduled_tokens
                    req_idx=req_idx+1
                    scheduled_seqs.append(seq)
            else:
                # chunked prefill
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                seq.num_scheduled_tokens = min(num_tokens, token_budget)
                token_budget= token_budget - seq.num_scheduled_tokens
                req_idx=req_idx+1
                scheduled_seqs.append(seq)

        if scheduled_seqs and token_budget == 0:
            return scheduled_seqs, True

        # waiting
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = self.waiting[0]
            assert not seq.block_table
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            self.waiting.popleft()
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, token_budget)
            token_budget= token_budget - seq.num_scheduled_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)

        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
