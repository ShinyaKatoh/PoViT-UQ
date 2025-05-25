import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary

class MCDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, training=True)
    
class DSconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        
        if stride == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, 
                                            out_channels=in_ch,
                                            kernel_size=kernel_size, 
                                            groups=in_ch, 
                                            padding='same')
        elif stride >= 2:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, 
                                            out_channels=in_ch,
                                            kernel_size=kernel_size, 
                                            groups=in_ch, 
                                            stride=stride,
                                            padding=(kernel_size - 1) // 2)
        
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, 
                                        out_channels=out_ch,
                                        kernel_size=1, 
                                        padding='same')
        
    def forward(self, x):
        
        x = rearrange(x, 'B L C -> B C L')
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        x = rearrange(x, 'B C L -> B L C')
       
        return  x


class InputLayer(nn.Module): 
    def __init__(self, in_channels, emb_dim, kernel_size, stride, in_length):
        super().__init__() 
        
        if stride == 1:
            
            self.conv = nn.Conv1d(in_channels = in_channels,
                                out_channels = emb_dim,
                                kernel_size = kernel_size,
                                padding = 'same',
                                bias=False)
            
        else:
        
            if kernel_size % 2 != 0: 

                self.conv = nn.Conv1d(in_channels = in_channels,
                                    out_channels = emb_dim,
                                    kernel_size = kernel_size,
                                    stride=stride,
                                    padding = kernel_size//2,
                                    bias=False)
                
            elif kernel_size % 2 != 1: 
                
                self.conv = nn.Conv1d(in_channels = in_channels,
                                    out_channels = emb_dim,
                                    kernel_size = kernel_size,
                                    stride=stride,
                                    padding = kernel_size//2-1,
                                    bias=False)
        
        

        # クラストークン 
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # 位置埋め込み
        self.pos_emb = nn.Parameter(torch.randn(1, in_length//stride+1, emb_dim))

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = rearrange(x, 'B C L -> B L C')
        x = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), x], dim=1)
        x = x + self.pos_emb
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 channels,
                 num_heads,
                 kernel_size,
                 stride,
                 dropout_ratio):
        super().__init__()
       
        self.linear_q = DSconv(channels, channels, kernel_size=kernel_size, stride=1)
        self.linear_k = DSconv(channels, channels, kernel_size=kernel_size, stride=stride)
        self.linear_v = DSconv(channels, channels, kernel_size=kernel_size, stride=stride)
        
        self.ln_q = nn.LayerNorm(channels)
        self.ln_k = nn.LayerNorm(channels)
        self.ln_v = nn.LayerNorm(channels)
        
        self.head = num_heads
        self.head_ch = channels // num_heads
        self.sqrt_dh = self.head_ch**0.5 
        
        self.attn_drop = MCDropout(dropout_ratio)

        self.w_o = nn.Linear(channels, channels, bias=False)
        self.w_drop = MCDropout(dropout_ratio)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        
        cls_token = x[:,:1]
        x = x[:,1:]
    
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
            
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)
        
        q = torch.cat((cls_token, q), dim=1)
        k = torch.cat((cls_token, k), dim=1)
        v = torch.cat((cls_token, v), dim=1)
            
        q = rearrange(q, 'B L (h C) -> B h L C', h=self.head)
        k = rearrange(k, 'B L (h C) -> B h L C', h=self.head)
        v = rearrange(v, 'B L (h C) -> B h L C', h=self.head)
        
        k_T = k.transpose(2, 3)
        
        dots = (q @ k_T) / self.sqrt_dh
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        out = attn @ v
        
        out = rearrange(out, 'B h L C -> B L (h C)')
        
        out = self.w_o(out) 
        out = self.w_drop(out)
        
        return out, attn
    
class MixFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 kernel_size,
                 dropout_ratio,
                 expantion_ratio:int=4):
        super().__init__()
        self.linear1 = nn.Conv1d(emb_dim, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.linear2 = nn.Conv1d(emb_dim * expantion_ratio, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.conv = nn.Conv1d(in_channels=emb_dim, 
                              out_channels=emb_dim * expantion_ratio, 
                              kernel_size=kernel_size, 
                              groups=emb_dim,
                              padding='same')
        
        self.gelu = nn.GELU()
        
        self.drop = MCDropout(dropout_ratio)

    def forward(self, x):
     
        x1 = x[:,:1]
        x2 = x[:,1:]
        
        x2 = rearrange(x2, 'B L C -> B C L')
        
        x2 = self.linear1(x2)
    
        x2 = self.conv(x2)
        x2 = self.gelu(x2)
        x2 = self.drop(x2)
        
        x2 = self.linear2(x2)
       
        x2 = rearrange(x2, 'B C L -> B L C')
        
        out = torch.cat((x1, x2), dim=1)
        
        return out
    
class MHSAtoFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 head_num,
                 ds_kernel_size,
                 ff_kernel_size,
                 stride,
                 dropout_ratio):
        
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(emb_dim,
                                           head_num,
                                           ds_kernel_size,
                                           stride,
                                           dropout_ratio)
        
        # self.ffn = FFN(emb_dim, 
        #                dropout_ratio)
        
        self.ffn = MixFFN(emb_dim, 
                          ff_kernel_size,
                          dropout_ratio)
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
       
    def forward(self, x):
        
        # print(x.shape)
        
        residual_mhsa = x
        mhsa_input = self.ln1(x)
        mhsa_output, attn = self.mhsa(mhsa_input)
        # print(mhsa_output.shape, attn.shape)
        mhsa_output2 = mhsa_output + residual_mhsa
        
        residual_ffn = mhsa_output2
        ffn_input = self.ln2(mhsa_output2)
        ffn_output = self.ffn(ffn_input) + residual_ffn
        
        return ffn_output
    
class ViTEncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim,
                 head_num,
                 ds_kernel_size,
                 ff_kernel_size,
                 stride,
                 dropout_ratio,
                 block_num):
        super().__init__()
       
        self.Encoder = nn.Sequential(*[MHSAtoFFN(emb_dim,
                                                 head_num,
                                                 ds_kernel_size,
                                                 ff_kernel_size,
                                                 stride,
                                                 dropout_ratio)
                                       for _ in range(block_num)])
        
    def forward(self, x):
        x = self.Encoder(x)
        return x
    
class Classification(nn.Module):
    def __init__(self,
                 emb_dim,
                 dropout_ratio,
                 class_num):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, class_num)
        self.drop = MCDropout(dropout_ratio)
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
class Segmentation(nn.Module):
    def __init__(self, 
                 ch,
                 kernel_size,
                 class_num,
                 dropout_ratio,
                 upr):
        super().__init__()
        
        self.upr = upr
        
        self.conv1 = nn.Conv1d(in_channels=ch, 
                               out_channels=ch, 
                               kernel_size=kernel_size,
                               padding='same')
        
        self.conv2 = nn.Conv1d(in_channels=ch,
                               out_channels=class_num,
                               kernel_size=kernel_size,
                               padding='same')
        
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
        
        self.drop = MCDropout(dropout_ratio)
        
        if self.upr > 1:
            self.up1 = nn.Upsample(scale_factor=upr, mode='linear', align_corners=True)

        
    def forward(self, x):
        
        if self.upr > 1:
            x = self.up1(x)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.conv2(x)
        x = self.softmax(x)
        
        return x
    
class Model(nn.Module): 
    def __init__(self, 
                 in_channels:int=1, 
                 class_num1:int=3, 
                 class_num2:int=2, 
                 emb_dim:int=128, 
                 in_length:int=256,
                 kernel_size:int=3,
                 ds_kernel_size:int=7,
                 ff_kernel_size:int=3,
                 seg_kernel_size:int=7,
                 stride:int=4,
                 dsstride:int=2,
                 num_blocks:int=5, 
                 head_num:int=4, 
                 dropout_ratio:float=0.2):
        super().__init__()
        self.input = InputLayer(in_channels, 
                                emb_dim, 
                                kernel_size, 
                                stride,
                                in_length)

        self.encoder = ViTEncoderBlock(emb_dim,
                                       head_num,
                                       ds_kernel_size,
                                       ff_kernel_size,
                                       dsstride,
                                       dropout_ratio,
                                       num_blocks)
         
        self.cla = Classification(emb_dim, dropout_ratio, class_num1)
        
        self.seg = Segmentation(emb_dim, 
                                seg_kernel_size,
                                class_num2,
                                dropout_ratio,
                                stride)

    def forward(self, x):
        out = self.input(x)
        out = self.encoder(out)

        cls_token = out[:,0]
        seg_token = rearrange(out[:,1:], 'B L C -> B C L')
        
        pred_cla = self.cla(cls_token)
        pred_seg = self.seg(seg_token)
        
        return pred_cla, pred_seg

    
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model = Model(in_length=512, kernel_size=5, stride=2, emb_dim=64).to('cpu')
    
    # sys.path.append('ViT_MixFNN_DSconv_0709')
    # from model_ViT import Model as ModelClass1
    # sys.path.pop()
    device = torch.device("cpu")
    emb_dim = 64
    kernel_size = 16
    ds_kernel_size= 9
    ff_kernel_size= 9
    seg_kernel_size= 9
    stride = 1
    length = 256
    block_num = 7
    head_num = 4
    dropout_ratio = 0.3
    model = Model(in_length=length, kernel_size=kernel_size, ds_kernel_size=ds_kernel_size, ff_kernel_size=ff_kernel_size, seg_kernel_size=seg_kernel_size, stride=stride, head_num=head_num, emb_dim=emb_dim, num_blocks=block_num, dropout_ratio=dropout_ratio)
    
    summary(model, input_size=(32, 1, 256))
