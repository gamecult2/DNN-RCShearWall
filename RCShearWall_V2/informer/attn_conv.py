import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, dim, E = K.shape
        _, _, L_Q, dim, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-4).expand(B, H, L_Q, L_K, dim, E)
        #print("K_expand", K_expand.shape)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        #print("is", index_sample.shape)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        #print("ksample", K_sample.shape, Q.unsqueeze(-2).shape, K_sample.permute(0,1,2,4,5,3).shape)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.permute(0,1,2,4,5,3)).squeeze()
        #print("Q_K_sample", Q_K_sample.shape)
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        #print("M", M.shape)
        M_top = M.topk(n_top, dim = 2, sorted=False)[1]
        #print("Mtop", M_top.shape)
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None, None],
                     torch.arange(H)[None, :, None, None],
                     M_top, torch.arange(dim)[None, None, None, :], :]
        
        #print("Q_reduce", Q_reduce.permute(0,1,3,2,4).shape, K.shape)
        Q_K = torch.matmul(Q_reduce.permute(0,1,3,2,4), K.permute(0,1,3,4,2)) 
        print("Q_K", Q_K.shape)
        #Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        print("V", V.shape)
        B, H, L_V, dim, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-3)
            contex = V_sum.unsqueeze(-3).expand(B, H, L_Q, dim, V_sum.shape[-1]).clone()
            print("contex", contex.shape)
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, dim, D = V.shape
        context_in = context_in.permute(0,1,3,2,4)
        V = V.permute(0,1,3,2,4)
        index = index.permute(0,1,3,2)
        
        #print("context_in begore", context_in.shape)
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        #print("score", scores.shape)
        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        #print("no type", torch.matmul(attn, V).shape)
        #print("index", index.shape)
        context_in[torch.arange(B)[:, None, None, None],
                   torch.arange(H)[None, :, None, None],
                   torch.arange(dim)[None, None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        print("context_in", context_in.shape)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, dim, H, D,  = queries.shape
        _, L_K, dim, _, _ = keys.shape

        #queries = queries.transpose(2,1)
        #keys = keys.transpose(2,1)
        #values = values.transpose(2,1)
        queries = queries.permute(0,3,1,2,4)
        keys = keys.permute(0,3,1,2,4)
        values = values.permute(0,3,1,2,4)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        #print(values.shape)
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        print("context, attn", context.shape)
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        #print(queries.shape, keys.shape)
        B, L, dim, _ = queries.shape
        _, S, dim, _ = keys.shape
        H = self.n_heads
        print("queries", queries.shape, "keys", keys.shape)
        queries = self.query_projection(queries).view(B, L, dim, H, -1)
        keys = self.key_projection(keys).view(B, S, dim, H, -1)
        values = self.value_projection(values).view(B, S, dim, H, -1)
        print("after queries", queries.shape, "keys", keys.shape, values.shape)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        #out = out.view(B, L, dim, -1)
        out = out.view(B, L, dim, -1)
        return self.out_projection(out), attn