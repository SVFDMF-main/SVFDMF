import torch
from torch import nn
from untils.tools import pretrain_bert_models
from model.Transformer import Transformer
from model.mambaformer import Mambaformer

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, dim))
        self.scale = dim ** -0.5

    def forward(self, x):
        attn = (x @ self.query.T) * self.scale
        attn = torch.softmax(attn, dim=-2)
        pooled = (attn * x).sum(dim=-2)
        return pooled
class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.adaptive_max_pool_layer = nn.AdaptiveMaxPool1d(output_size=1)
        
    def forward(self, fea):
        pooled_sequence = self.adaptive_max_pool_layer(fea.transpose(1, 2)).transpose(1, 2).squeeze(dim=-2)
        return pooled_sequence

class SVMFModel(nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(SVMFModel, self).__init__()

        self.bert = pretrain_bert_models()
        self.text_dim = 1024
        self.video_dim = 1024
        self.hubert_dim = 1024
        self.dim = fea_dim
        self.num_heads = 16
        self.trans_dim = 512
        self.dropout = dropout
        
        self.maxpool = MaxPool()
        
        # 增加 Transformer 和 Mambaformer 层数
        self.mambaformer = Mambaformer(input_dim=512, model_dimension=self.trans_dim, num_layers=self.num_heads,
                                        number_of_heads=self.num_heads, dropout_probability=self.dropout)

        self.linear_text = nn.Sequential(nn.Linear(self.text_dim, self.trans_dim), nn.ReLU(), nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(nn.Linear(self.video_dim, self.trans_dim), nn.ReLU(), nn.Dropout(p=self.dropout))
        self.linear_hubert = nn.Sequential(nn.Linear(self.hubert_dim, self.trans_dim), nn.ReLU(), nn.Dropout(p=self.dropout))
        #模态自注意力引入 先强化单模态内部关系，再送mambaformer,再进行跨模态交互。
        self.self_attn_text = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        self.self_attn_audio = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        self.self_attn_video = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        
       # 跨模态交互层
        self.ta_trm = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        self.tv_trm = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        self.va_trm = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)

        # Attention Pooling 代替 Max Pooling
        self.attn_pool_ta = AttentionPooling(self.trans_dim)
        self.attn_pool_tv = AttentionPooling(self.trans_dim)
        self.attn_pool_va = AttentionPooling(self.trans_dim)

        # 自适应融合网络（MLP层）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.trans_dim * 3, self.trans_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.trans_dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )


        # 分类器
        self.classifier = nn.Linear(fea_dim, 2)

    def forward(self, **kwargs):
        title_inputid = kwargs['title_inputid']
        title_mask = kwargs['title_mask']
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']
        fea_text = self.linear_text(fea_text)

        audio_feas = kwargs['audio_feas']
        fea_audio = self.linear_hubert(audio_feas)

        frames = kwargs['frames']
        fea_video = self.linear_video(frames)
       
        # 自transformer
        fea_text = self.self_attn_text(fea_text, fea_text)
        fea_audio = self.self_attn_audio(fea_audio, fea_audio)
        fea_video = self.self_attn_video(fea_video, fea_video)

        fea_text,fea_audio,fea_video = self.mambaformer(fea_text,fea_audio,fea_video)
        
        # 跨模态交互
        fea_ta = self.ta_trm(fea_text, fea_audio)
        fea_tv = self.tv_trm(fea_text, fea_video)
        fea_va = self.va_trm(fea_video, fea_audio)
        
        # Attention Pooling
       
        fea_ta = self.attn_pool_ta(fea_ta)
        fea_tv = self.attn_pool_tv(fea_tv)
        fea_va = self.attn_pool_va(fea_va)
        
        # 将三个模态特征拼接后通过MLP自适应融合
        fea_concat = torch.cat([fea_ta, fea_tv, fea_va], dim=-1)
        fea_fused = self.fusion_mlp(fea_concat)

        # 分类器
        output = self.classifier(fea_fused)
        return output, fea_fused
