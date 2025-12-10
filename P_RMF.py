import torch
from torch import nn
from .basic_layers import Transformer, GradientReversalLayer, CrossmodalEncoder
from .bert import BertTextEncoder
from einops import rearrange, repeat
from .generate_proxy_modality import Generate_Proxy_Modality
import torch.nn.functional as F


class P_RMF(nn.Module):
    def __init__(self, args):
        super(P_RMF, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True,
                                         transformers=args['model']['feature_extractor']['transformers'],
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                      args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                      args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )
        self.generate_proxy_modality = Generate_Proxy_Modality(args, args['model']['generate_proxy']['input_dim'],
                                                               args['model']['generate_proxy']['hidden_dim'],
                                                               args['model']['generate_proxy']['out_dim'])

        self.GRL = GradientReversalLayer(alpha=1.0)
        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['reconstructor']['input_dim'],
                        depth=args['model']['reconstructor']['depth'],
                        heads=args['model']['reconstructor']['heads'],
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])

        self.crossmodal_encoder = CrossmodalEncoder(proxy_dim=args['model']['crossmodal_encoder']['proxy_dim'],
                                                    text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0],
                                                    audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2],
                                                    video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1],
                                                    embed_dim=args['model']['crossmodal_encoder']['embed_dim'],
                                                    num_layers=args['model']['crossmodal_encoder']['num_layers'],
                                                    attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])

        self.fc1 = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['hidden_dim'])
        self.fc2 = nn.Linear(args['model']['regression']['hidden_dim'], args['model']['regression']['out_dim'])
        self.dropout = nn.Dropout(args['model']['regression']['attn_dropout'])

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        complete_language_feat, complete_vision_feat, complete_audio_feat = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

        # Proxy dominate modality generator
        kl_loss, proxy_m, weight_t_v_a = self.generate_proxy_modality(h_1_l, h_1_v, h_1_a, complete_language_feat,
                                                                      complete_vision_feat, complete_audio_feat)

        proxy_m = self.GRL(proxy_m)
        feat = self.crossmodal_encoder(proxy_m, h_1_l, h_1_a, h_1_v, weight_t_v_a)

        output = self.predict(torch.mean(feat, dim=1))

        rec_feats, complete_feats, effectiveness_discriminator_out, proxy_X, kl_p = None, None, None, None, 0.0
        if (vision is not None) and (audio is not None) and (language is not None):
            # Reconstruction
            rec_feat_a = self.reconstructor[0](h_1_a)[:, :8]
            rec_feat_v = self.reconstructor[1](h_1_v)[:, :8]
            rec_feat_l = self.reconstructor[2](h_1_l)[:, :8]
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat],
                                       dim=1)

        return {'sentiment_preds': output,
                'rec_feats': rec_feats,
                'complete_feats': complete_feats,
                'kl_loss': kl_loss
                }


def build_model(args):
    return P_RMF(args)
