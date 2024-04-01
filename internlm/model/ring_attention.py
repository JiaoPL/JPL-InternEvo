import torch
import torch.nn as nn

from einops import rearrange

from ring_flash_attn import zigzag_ring_flash_attn_qkvpacked_func, zigzag_ring_flash_attn_varlen_qkvpacked_func, zigzag_ring_flash_attn_kvpacked_func, zigzag_ring_flash_attn_varlen_kvpacked_func , ring_flash_attn_kvpacked_func, ring_flash_attn_qkvpacked_func, ring_flash_attn_varlen_kvpacked_func, ring_flash_attn_varlen_qkvpacked_func

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger
logger = get_logger(__file__)

def split_seqlens(cu_seqlens, total_slices, each_seqlen, return_idx=None):
    cu_seqlens_splits = []
    current_split_start = 0
    
    for i in range(1, total_slices + 1):
        current_split_end = i * each_seqlen
        split_indices = (cu_seqlens >= current_split_start) & (cu_seqlens < current_split_end)
        current_split = cu_seqlens[split_indices]
        if len(current_split) == 0 or current_split[0] %  each_seqlen!= 0:
            current_split = torch.cat((torch.tensor([current_split_start], device=cu_seqlens.device), current_split))

        current_split = torch.cat((current_split, torch.tensor([current_split_end], device=cu_seqlens.device)))
        if i == total_slices:
            if current_split_end != cu_seqlens[-1]:
                current_split = torch.cat((current_split, cu_seqlens[-1:]))
        
        cu_seqlens_splits.append(current_split)
        current_split_start = current_split_end
    
    if return_idx is not None:
        new_cu_seqlens = cu_seqlens_splits[return_idx]
        new_cu_seqlens = new_cu_seqlens - each_seqlen * return_idx
        new_cu_seqlens = new_cu_seqlens.to(torch.int32)
        max_seqlen = max([new_cu_seqlens[i+1] - new_cu_seqlens[i] for i in range(len(new_cu_seqlens)-1)])
        
        return new_cu_seqlens, max_seqlen.item()

    split_cu_seqlens_and_max = []
    for idx in range(total_slices):
        new_cu_seqlens = cu_seqlens_splits[idx]
        new_cu_seqlens = new_cu_seqlens - each_seqlen * idx
        new_cu_seqlens = new_cu_seqlens.to(torch.int32)
        max_seqlen = max([new_cu_seqlens[i+1] - new_cu_seqlens[i] for i in range(len(new_cu_seqlens)-1)])
        split_cu_seqlens_and_max.append((new_cu_seqlens, max_seqlen.item()))
    return split_cu_seqlens_and_max



class RingFlashSelfAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.group = gpc.get_group(ParallelMode.TENSOR)


    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None

        if unpadded:
            seqlen = qkv.shape[0]
            rank = gpc.get_local_rank(ParallelMode.TENSOR)
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            cu_seqlens, max_seqlen = split_seqlens(cu_seqlens,world_size, seqlen, return_idx=rank)
            # logger.info(f"cuda memory profiling: max_allocated {torch.cuda.max_memory_allocated()}, allocated {torch.cuda.memory_allocated()}")
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            if gpc.config.data.pack_sample_into_one:
                return zigzag_ring_flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen, self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )
            else:
                return ring_flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen, self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )

        else:
            if gpc.config.data.pack_sample_into_one:
                return zigzag_ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                            softmax_scale=self.softmax_scale, causal=causal, group=self.group)
            else:
                return ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                            softmax_scale=self.softmax_scale, causal=causal, group=self.group)



class RingFlashCrossAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.group = gpc.get_group(ParallelMode.TENSOR)


    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
                cu_seqlens_k=None, max_seqlen_k=None):
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None

        if unpadded:
            seqlen = q.shape[0]
            rank = gpc.get_local_rank(ParallelMode.TENSOR)
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            cu_seqlens, max_seqlen = split_seqlens(cu_seqlens,world_size, seqlen, return_idx=rank)
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            if gpc.config.data.pack_sample_into_one:
                return zigzag_ring_flash_attn_varlen_kvpacked_func(
                    q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )
            else:
                return ring_flash_attn_varlen_kvpacked_func(
                    q, kv,cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            if gpc.config.data.pack_sample_into_one:
                return zigzag_ring_flash_attn_kvpacked_func(q, kv, self.drop.p if self.training else 0.0,
                                            causal=causal, softmax_scale=self.softmax_scale, group=self.group)
            else:
                return ring_flash_attn_kvpacked_func(q, kv, self.drop.p if self.training else 0.0,
                                            causal=causal, softmax_scale=self.softmax_scale, group=self.group)


def blockwise_ffn(remat_ffn, inputs, chunk_size, batch_size):
    ndims = inputs.dim()
    if ndims == 2:
        # train
        inputs = rearrange(inputs, '(b c n) hd -> b c n hd', c=chunk_size, b = batch_size)
    else:
        # evaluation
        inputs = rearrange(inputs, 'b (c n) hd -> b c n hd', c=chunk_size)
    num_chunks = inputs.shape[2]
    outs = []
    
    for i in range(num_chunks):
        out = remat_ffn(inputs[:,:,i,:])
        outs.append(out.unsqueeze(2))

    outs = torch.cat(outs, dim=2)
    if ndims == 2:
        outs = rearrange(outs, 'b c n hd -> (b c n) hd')
    else:
        outs = rearrange(outs, 'b c n hd -> b (c n) hd')

    return outs

