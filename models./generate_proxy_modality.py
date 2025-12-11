import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from .basic_layers import Transformer


class VariationalEncoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()

        self.encoder = nn.Sequential(
            Transformer(num_frames=args['model']['vae']['input_length'], #8
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['vae']['input_dim'],  #128
                        depth=args['model']['vae']['depth'],  #2
                        heads=args['model']['vae']['heads'], #8
                        mlp_dim=args['model']['vae']['hidden_dim']) #128
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim) #(128,128)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  #(128,128) # Mean
        self.fc2_log_var = nn.Linear(hidden_dim, latent_dim) #(128,128)  # Log variance

    def _initialize_weights(self):
        # Use Xavier initialization for linear layer weights
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x):   # x为不完整输入:(B,8,128)
        h = torch.relu(self.fc1(x))   #(B,8,128), 特征维从 input_dim 投到 hidden_dim
        memory = self.encoder(h)#(B,8,128),用一个Transformer在序列维做上下文建模
        # memory = memory.mean(dim=1)  # (batch_size, hidden_dim)
        # 再接线性层得到 μ / log σ²
        mu = self.fc2_mu(memory) #(B,8,128), 线性映射得到 μ(稳定语义)
        log_var = self.fc2_log_var(memory) #(B,8,128), 线性映射得到 log σ²（不确定性）
        # std = torch.exp(0.5 * log_var).clamp(min=1e-6)  # Ensure standard deviation is positive
        return mu, log_var

#对先验的 KL：
def s_kl_divergence(mu, log_var):
    assert torch.isfinite(mu).all(), "mu contains NaN or Inf"
    assert torch.isfinite(log_var).all(), "log_var contains NaN or Inf"
    # q,p :(16,8,128)
    q = Normal(mu, torch.exp(0.5 * log_var).clamp(min=1e-6)) #对 σ 做clamp(min=1e-6)，避免数值非正 # Ensure standard deviation is positive and non-zero
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))  # Standard normal distribution N(0, I)
    kl = kl_divergence(q, p) #(16,8,128)
    # mean 会在 batch × T × d 三个维度上平均，变为一个数！
    kl_mean = torch.mean(kl) 
    return kl_mean


# Decoder: generates reconstructed data from latent space samples
class Decoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)#(128,128)

        self.decoder = nn.Sequential(
            Transformer(num_frames=args['model']['vae']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['vae']['input_dim'],
                        depth=args['model']['vae']['depth'],
                        heads=args['model']['vae']['heads'],
                        mlp_dim=args['model']['vae']['hidden_dim'])
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))#(B,8,128) 
        output = self.decoder(h) #(B,8,128) 
        output = self.fc_out(output) #(B,8,128)  # [batch_size, seq_len, output_dim]
        # return torch.sigmoid(self.fc2(h))
        return output

    # VAE model: combines encoder and decoder


class VAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim): #128,128,128
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(args, input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(args, latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x) # Encoder 学 均值μ / 对数方差 log σ²
        # 重参数化采样：z=μ+σ⋅ϵ
        z = self.reparameterize(mu, log_var) #(16,8,128)
        # Decoder 重建（样本/特征级重建）
        x_recon = self.decoder(z) #(16,8,128)
        return x_recon, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) #(B,8,128)  标准差 σ
        eps = torch.randn_like(std) # ε 采样随机, # ϵ ~ N(0, I)
        z = mu + std * eps
        return z  #(B,8,128) 

# x, x_recon：(16,8,128)
def recon_loss(x, x_recon):
    mse_loss = nn.MSELoss() #一个数！！！一个批次(16个样本)的损失值取平均
    # return nn.functional.binary_cross_entropy(x_recon, x, reduction='mean')
    return mse_loss(x_recon, x)

#单模态VAE 损失(单模态先验KL损失 + 重建MSE损失)
def vae_loss(x, x_recon, mu, log_var):
    recon_loss_value = recon_loss(x, x_recon) #默认mean（会在 batch × T × d 三个维度上平均）
    kl_loss_value = s_kl_divergence(mu, log_var)
    return recon_loss_value + kl_loss_value


# 为每个模态设计一个变分自编码器（VAE）,每个VAE 包括一个编码器 VariationalEncoder 和解码器Decoder
# 把单模态特征映射到高斯潜空间
class Generate_Proxy_Modality(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim): # 128,128,128
        super(Generate_Proxy_Modality, self).__init__()

        self.text_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.image_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.audio_VAE = VAE(args, input_dim, hidden_dim, latent_dim)

    def forward(self, text, video, audio, c_text, c_vision, c_audio):

        t_recon, mu_t, log_var_t = self.text_VAE(text) #(B,8,128) 
        v_recon, mu_v, log_var_v = self.image_VAE(video) #(B,8,128) 
        a_recon, mu_a, log_var_a = self.audio_VAE(audio) #(B,8,128) 
        
        # *训练信号1：单模态 VAE 损失（对先验的KL + 重建MSE）
        loss_t, loss_v, loss_a = 0, 0, 0
        # 是否用 完整特征 做重建监督  →  是否传入了 c_text / c_vision / c_audio
        #给了完整数据 → 用完整数据监督
        if (c_text is not None) and (c_vision is not None) and (c_audio is not None):
            loss_t = vae_loss(c_text, t_recon, mu_t, log_var_t)
            loss_a = vae_loss(c_audio, a_recon, mu_a, log_var_a)
            loss_v = vae_loss(c_vision, v_recon, mu_v, log_var_v)
        else: #否则 → 用输入自身做重建
            loss_t = vae_loss(text, t_recon, mu_t, log_var_t)
            loss_a = vae_loss(audio, a_recon, mu_a, log_var_a)
            loss_v = vae_loss(video, v_recon, mu_v, log_var_v)
        #*训练信号2：跨模态对齐 KL（让不同模态的后验分布彼此对齐,降低分布偏差）
        # std_m，由对数方差计算 标准差
        std_t = torch.exp(0.5 * log_var_t)
        std_v = torch.exp(0.5 * log_var_v)
        std_a = torch.exp(0.5 * log_var_a)

        qv = Normal(mu_v, std_v)
        qt = Normal(mu_t, std_t)
        qa = Normal(mu_a, std_a)
        #把三个模态的高斯后验两两配对, 调用 kl_divergence，再取均值
        kl_v_t = kl_divergence(qv, qt).mean()
        kl_a_t = kl_divergence(qa, qt).mean()
        kl_a_v = kl_divergence(qa, qv).mean()
        #把三对跨模态KL损失与三模态各自的VAE 损失（KL+SR）合在一起，取平均得到 总变分约束 kl_loss
        kl_loss = (kl_v_t + kl_a_t + kl_a_v + loss_t + loss_a + loss_v) / 3
        

        # 模态不确定性权重 ω, 方差小 → 更“确定”/可信 → 权重应更大. 
        std_i_m = torch.stack([std_t, std_v, std_a], dim=0) #堆成 [3, B, 8, 128] 
        weight_m = torch.exp(1 / std_i_m) / torch.sum(torch.exp(1 / std_i_m),
                                                      dim=0)# [3, B, 8, 128]  #在模态维做 softmax 归一 # Normalize along the modality dimension
        mu_i_m = torch.stack([mu_t, mu_v, mu_a], dim=0) # [3, B, 8, 128]

        #用不确定性导出的权重对各模态均值做和，得到稳定且可泛化的 Hₚ
        # 三个模态的 μ（稳定语义）按 ω 做加权和，沿模态维求和,得到代理模态
        proxy_m = torch.sum(weight_m * mu_i_m, dim=0) #[B, 8, 128]

        return kl_loss, proxy_m, weight_m


if __name__ == '__main__':
    # Assume input dimension and latent dimension
    input_dim = 10
    hidden_dim = 5
    latent_dim = 5
    num_multiway = 3  # Assume three modalities

    # Create model
    model = Generate_Proxy_Modality(input_dim, hidden_dim, latent_dim)

    # Generate some random data as input (batch_size, input_dim)
    batch_size = 4
    text_input = torch.randn(batch_size, input_dim)
    video_input = torch.randn(batch_size, input_dim)
    audio_input = torch.randn(batch_size, input_dim)

    # Test model
    kl_loss, proxy_m = model(text_input, video_input, audio_input)

    print("KL Loss:", kl_loss)
    print("Proxy_m:", proxy_m)
