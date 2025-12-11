import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F


# 单步跨模态注意力, 对应式(7)
class CrossModalAttention(nn.Module):
    def __init__(self, modality1_dim, modality2_dim, embed_dim, attn_dropout=0.5):
        super(CrossModalAttention, self).__init__()
        self.modality1_dim = modality1_dim
        self.modality2_dim = modality2_dim
        self.embed_dim = embed_dim
        self.modality1_ln = nn.LayerNorm(self.modality1_dim)
        self.modality2_ln = nn.LayerNorm(self.modality2_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scaling = self.embed_dim ** -0.5
        self.proj_modality1 = nn.Linear(self.modality1_dim, self.embed_dim)
        self.proj_modality2_k = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj_modality2_v = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.modality1_dim)
    # LN → 线性投影 → 缩放点积注意力 → 加权求和 → 映回 proxy 维度
    def forward(self, modality1, modality2):
        # 把两端都投到 embed_dim
        q = self.proj_modality1(self.modality1_ln(modality1))  #modality1:代理模态（proxy）,Qp
        k = self.proj_modality2_k(self.modality2_ln(modality2)) #modality2:文本/音频/视觉 的某一模态.Km
        v = self.proj_modality2_v(self.modality2_ln(modality2)) #Vm
        attention = F.softmax(torch.bmm(q, k.permute(0, 2, 1)) * self.scaling, dim=-1)
        context = torch.bmm(attention, v)
        output = self.proj(context)
        # output = self.attn_dropout(output)
        # output = output + self.modality1_ln(modality1)
        # output = output + modality1
        return output

# 单层 PDDI,式 (8) + (9)
class CrossmodalEncoderLayer(nn.Module):
    def __init__(self, proxy_dim,text_dim, audio_dim, video_dim, embed_dim, attn_dropout=0.5):
        super(CrossmodalEncoderLayer, self).__init__()
        self.cma_t = CrossModalAttention(proxy_dim, text_dim, embed_dim)
        self.cma_a = CrossModalAttention(proxy_dim, audio_dim, embed_dim)
        self.cma_v = CrossModalAttention(proxy_dim, video_dim, embed_dim)
        self.layernorm = nn.LayerNorm(proxy_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc = nn.Linear(proxy_dim, proxy_dim)

    def forward(self, proxy_m, text, audio, video,weight_t_v_a):
        if proxy_m is None:
            output = self.cma_a(text, audio) + self.cma_v(text, video) + text
        else: #对应式(8), 按模态权重加权求和，再与上一层 proxy 融合,output为Ul
            output = (self.cma_t(proxy_m, text) * weight_t_v_a[0] +  # ω^t · U^{pt}
                      self.cma_a(proxy_m, audio) * weight_t_v_a[2] + # ω^a · U^{pa}
                      self.cma_v(proxy_m, video) * weight_t_v_a[1] + # ω^v · U^{pv}
                      proxy_m) # + 残差 proxy 
        # output = self.cma_a(text, audio) + self.cma_v(text, video) + self.layernorm(text)
        # output = self.cma_a(proxy_m, text) +self.cma_a(proxy_m, audio) + self.cma_v(proxy_m, video) + proxy_m
        residual = output
        output = self.fc(self.layernorm(output)) #层归一 + 前馈
        output = self.attn_dropout(output)
        output = output + residual
        return output


class CrossmodalEncoder(nn.Module):
    def __init__(self, proxy_dim,text_dim, audio_dim, video_dim, embed_dim, num_layers=4, attn_dropout=0.5):
        super(CrossmodalEncoder, self).__init__()
        self.encoderlayer = CrossmodalEncoderLayer(proxy_dim,text_dim, audio_dim, video_dim, embed_dim, attn_dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        for layer in range(self.num_layers):  # 4层CrossmodalEncoderLayer !!!
            new_layer = self.encoderlayer
            self.layers.append(new_layer)
        # self.layers = nn.ModuleList([
        #   CrossmodalEncoderLayer(proxy_dim, text_dim, audio_dim, video_dim, embed_dim, attn_dropout)
        #   for _ in range(num_layers)
        # ])


    def forward(self, proxy_m, text, audio, video,weight_t_v_a):
        if proxy_m is None:
            output = text
            for layer in self.layers:
                output = layer(None, text, audio, video,weight_t_v_a)
            return output
        output = proxy_m
        # 每一层都重新以 proxy_m 为 Query，从三模态注入信息、做加权与残差，再进入下一层，逐步“在稳定核心上叠加多样细节
        for layer in self.layers:
            output = layer(output, text, audio, video,weight_t_v_a) #（16,8,128）
        return output




class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# 对抗式的稳定化
class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn1(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt



class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x



class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len #8
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):  #T:(B,500,128)
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)
        # x:(B,8,128)
        return x
