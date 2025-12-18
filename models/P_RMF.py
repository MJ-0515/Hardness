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
        # models/bert.py------------------------BertTextEncoder (True, bert, bert/bert-base-uncased)
        self.bertmodel = BertTextEncoder(use_finetune=True,
                                         transformers=args['model']['feature_extractor']['transformers'],
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], #nn.Linear(768,128)
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],  #basic_layers/ class Transformer
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'], #2
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )  #Transformer(num_frames=768,save_hidden=False,token_len=8, dim=128, depth=2, heads=8,mlp_dim= 128 )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], #nn.Linear(5,128)
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
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], #nn.Linear(20,128)
                      args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )
        #----------代理模态生成模块-----------Generate_Proxy_Modality(128,128,128)
        self.generate_proxy_modality = Generate_Proxy_Modality(args, args['model']['generate_proxy']['input_dim'],
                                                               args['model']['generate_proxy']['hidden_dim'],
                                                               args['model']['generate_proxy']['out_dim'])

        self.GRL = GradientReversalLayer(alpha=1.0) ##????
        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'], #8
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['reconstructor']['input_dim'], #128
                        depth=args['model']['reconstructor']['depth'], #2
                        heads=args['model']['reconstructor']['heads'], #8
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)  #128
        ])
        # basic_layers.py: class CrossmodalEncoder
        self.crossmodal_encoder = CrossmodalEncoder(proxy_dim=args['model']['crossmodal_encoder']['proxy_dim'], #128
                                                    text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0], #128
                                                    audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2], #128
                                                    video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1], #128
                                                    embed_dim=args['model']['crossmodal_encoder']['embed_dim'],  #128
                                                    num_layers=args['model']['crossmodal_encoder']['num_layers'],  #4
                                                    attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])  #0.5

        self.fc1 = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['hidden_dim']) #(128,128)
        self.fc2 = nn.Linear(args['model']['regression']['hidden_dim'], args['model']['regression']['out_dim']) #(128,1)
        self.dropout = nn.Dropout(args['model']['regression']['attn_dropout'])  #0.5

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output
     #  V:(16,500,20)    A:(16,375,5)  T(16,3,50); V_m:(16,500,20)    A_m:(16,375,5)  T_m(16,3,50)
    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input #v:[B,500,20], a:[B,375,5],l:[B,3,50]
        #*Step 1: 不完整输入 统一投影到同一维度/长度
        h_1_v = self.proj_v(vision_m)[:, :8]  #[B,8,128]
        h_1_a = self.proj_a(audio_m)[:, :8] #[B,8,128]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8] #[B,8,128]

        complete_language_feat, complete_vision_feat, complete_audio_feat = None, None, None
        #*Step 2: 完整输入  进行投影（维度同上），传给 PMG 作为重建监督（完整三模都给到时才会开启重建损监督；否则只做预测）
        if (vision is not None) and (audio is not None) and (language is not None):
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

        #*Step 3: PMG 输出：kl损失, 代理模态 proxy_m [B,8,128], 模态不确定性权重 weight_t_v_a 
        # kl_loss, proxy_m, weight_t_v_a = self.generate_proxy_modality(h_1_l, h_1_v, h_1_a, complete_language_feat,
        #                                                               complete_vision_feat, complete_audio_feat)
        kl_loss, proxy_m, weight_t_v_a = self.generate_proxy_modality(h_1_l, h_1_v, h_1_a, complete_language_feat,
                                                                       complete_vision_feat, complete_audio_feat)

        #把主导模态过一次 Gradient Reversal Layer (GRL)，目的是在后续对抗/域不变学习里让代理模态更“稳健”（数值方向反转，使上游学到更判别/稳健的表征）
        proxy_m = self.GRL(proxy_m) 
        # proxy_m、三模态不完整输入、模态权重weight_t_v_a 一起送入 CrossmodalEncoder（即 PDDI 动态跨模态注入模块）
        feat = self.crossmodal_encoder(proxy_m, h_1_l, h_1_a, h_1_v, weight_t_v_a)#（16,8,128）
        #逐层用 proxy_m 作为 Query，对（t,a,v）做 CMA，并按 weight_t_v_a[0/2/1] 加权：
        #cma_t(proxy,t)*w_t + cma_a(proxy,a)*w_a + cma_v(proxy,v)*w_v + proxy
        #先对 token 做均值池化，再过 MLP(128→128→out_dim)得到情感预测值

        
        fusion_feat = torch.mean(feat, dim=1)  # [B, D]

        output = self.predict(fusion_feat) #(B,1)
        #(可选)重建分支输出监督信号
        rec_feats, complete_feats, effectiveness_discriminator_out, proxy_X, kl_p = None, None, None, None, 0.0
        if (vision is not None) and (audio is not None) and (language is not None):
            # 各模用各自reconstructor[i] 对 不完整投影 h_1_* 进一步编码，截取 [:, :8]
            rec_feat_a = self.reconstructor[0](h_1_a)[:, :8] #(16,8,128)
            rec_feat_v = self.reconstructor[1](h_1_v)[:, :8]
            rec_feat_l = self.reconstructor[2](h_1_l)[:, :8]
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1) #(B,24,128)

            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat],
                                       dim=1) #(B,24,128)

        return {'sentiment_preds': output,
                'rec_feats': rec_feats,
                'complete_feats': complete_feats,
                'kl_loss': kl_loss,
                'h_1_l': h_1_l,
                'h_1_v': h_1_v,
                'h_1_a': h_1_a,
                'weight_t_v_a': weight_t_v_a
                }
 


def build_model(args):
    return P_RMF(args)
