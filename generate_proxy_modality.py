import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from .basic_layers import Transformer


class VariationalEncoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()

        self.encoder = nn.Sequential(
            Transformer(num_frames=args['model']['vae']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['vae']['input_dim'],
                        depth=args['model']['vae']['depth'],
                        heads=args['model']['vae']['heads'],
                        mlp_dim=args['model']['vae']['hidden_dim'])
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc2_log_var = nn.Linear(hidden_dim, latent_dim)  # Log variance

    def _initialize_weights(self):
        # Use Xavier initialization for linear layer weights
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        memory = self.encoder(h)
        # memory = memory.mean(dim=1)  # (batch_size, hidden_dim)

        mu = self.fc2_mu(memory)
        log_var = self.fc2_log_var(memory)
        # std = torch.exp(0.5 * log_var).clamp(min=1e-6)  # Ensure standard deviation is positive
        return mu, log_var


def s_kl_divergence(mu, log_var):
    assert torch.isfinite(mu).all(), "mu contains NaN or Inf"
    assert torch.isfinite(log_var).all(), "log_var contains NaN or Inf"

    q = Normal(mu, torch.exp(0.5 * log_var).clamp(min=1e-6))  # Ensure standard deviation is positive and non-zero
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))  # Standard normal distribution N(0, I)
    kl = kl_divergence(q, p)

    kl_mean = torch.mean(kl)
    return kl_mean


# Decoder: generates reconstructed data from latent space samples
class Decoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)

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
        h = torch.relu(self.fc1(z))
        output = self.decoder(h)
        output = self.fc_out(output)  # [batch_size, seq_len, output_dim]
        # return torch.sigmoid(self.fc2(h))
        return output

    # VAE model: combines encoder and decoder


class VAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(args, input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(args, latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z


def recon_loss(x, x_recon):
    mse_loss = nn.MSELoss()
    # return nn.functional.binary_cross_entropy(x_recon, x, reduction='mean')
    return mse_loss(x_recon, x)


# Total loss function (reconstruction loss + KL divergence)
def vae_loss(x, x_recon, mu, log_var):
    recon_loss_value = recon_loss(x, x_recon)
    kl_loss_value = s_kl_divergence(mu, log_var)
    return recon_loss_value + kl_loss_value


# Define a simple variational encoder for each modality
class Generate_Proxy_Modality(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(Generate_Proxy_Modality, self).__init__()

        self.text_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.image_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.audio_VAE = VAE(args, input_dim, hidden_dim, latent_dim)

    def forward(self, text, video, audio, c_text, c_vision, c_audio):

        t_recon, mu_t, log_var_t = self.text_VAE(text)
        v_recon, mu_v, log_var_v = self.image_VAE(video)
        a_recon, mu_a, log_var_a = self.audio_VAE(audio)
        loss_t, loss_v, loss_a = 0, 0, 0
        if (c_text is not None) and (c_vision is not None) and (c_audio is not None):
            loss_t = vae_loss(c_text, t_recon, mu_t, log_var_t)
            loss_a = vae_loss(c_audio, a_recon, mu_a, log_var_a)
            loss_v = vae_loss(c_vision, v_recon, mu_v, log_var_v)
        else:
            loss_t = vae_loss(text, t_recon, mu_t, log_var_t)
            loss_a = vae_loss(audio, a_recon, mu_a, log_var_a)
            loss_v = vae_loss(video, v_recon, mu_v, log_var_v)

        std_t = torch.exp(0.5 * log_var_t)
        std_v = torch.exp(0.5 * log_var_v)
        std_a = torch.exp(0.5 * log_var_a)

        qv = Normal(mu_v, std_v)
        qt = Normal(mu_t, std_t)
        qa = Normal(mu_a, std_a)

        kl_v_t = kl_divergence(qv, qt).mean()
        kl_a_t = kl_divergence(qa, qt).mean()
        kl_a_v = kl_divergence(qa, qv).mean()

        kl_loss = (kl_v_t + kl_a_t + kl_a_v + loss_t + loss_a + loss_v) / 3

        std_i_m = torch.stack([std_t, std_v, std_a], dim=0)  # Stack standard deviations along a new dimension
        weight_m = torch.exp(1 / std_i_m) / torch.sum(torch.exp(1 / std_i_m),
                                                      dim=0)  # Normalize along the modality dimension
        mu_i_m = torch.stack([mu_t, mu_v, mu_a], dim=0)
        proxy_m = torch.sum(weight_m * mu_i_m, dim=0)

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
