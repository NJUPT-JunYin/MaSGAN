import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
class PatchemBed(nn.Module):
    def __init__(self,in_channels,embed_dim,img_size_h,img_size_w,kernel_size,stride):
        super(PatchemBed, self).__init__()
        img_size=(img_size_h,img_size_w)
        patch_size=(img_size[0]//kernel_size[0],img_size[1]//kernel_size[1])
        self.conv=nn.Conv2d(in_channels,out_channels=embed_dim,kernel_size=kernel_size,stride=stride)
        self.num_patches=patch_size[0]*patch_size[1]
    def forward(self,x):
        x=self.conv(x)
        x=x.flatten(2).permute(0,2,1)
        return x
'''img=Image.open('C:/Users/QianXinZhi/Desktop/无噪.png')
plt.imshow(img)
plt.show()
if img.mode == 'RGBA':
    img = img.convert('RGB')
transform=transforms.Compose([transforms.Resize([256,512]),transforms.ToTensor()])
img=transform(img)
img=img.unsqueeze(0)'''
class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(Attention,self).__init__()
        self.qkv=nn.Linear(embed_dim,3*embed_dim)
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.dropout=nn.Dropout(0.2)
        self.linear=nn.Linear(embed_dim,embed_dim)
    def forward(self,x):
        qkv=self.qkv(x)
        qkv=qkv.reshape(qkv.shape[0],-1,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]#[batch,num_head,num_patch,head_dim]
        attn=q@k.transpose(-1,-2)*self.head_dim**(-0.5)
        attn=nn.Softmax(dim=-1)(attn)
        attn=self.dropout(attn)
        x=(attn@v).transpose(1,2).flatten(2)
        x=self.linear(x)
        x=self.dropout(x)
        return x
class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU):
        super(Mlp,self).__init__()
        if hidden_features is None:
            hidden_features=in_features
        out_features=out_features or in_features
        self.linear1 = nn.Linear(in_features,hidden_features)
        self.linear2 = nn.Linear(hidden_features,out_features)
        self.act_layer=act_layer()
        self.dropout=nn.Dropout(0.2)
    def forward(self, x):
        x = self.linear1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio,act_layer=nn.GELU):
        super(Block,self).__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.atten=Attention(embed_dim=embed_dim,num_heads=num_heads)
        self.drop1=nn.Dropout(0.2)
        self.norm2=nn.LayerNorm(embed_dim)
        self.mlp=Mlp(in_features=embed_dim,hidden_features=mlp_ratio*embed_dim,act_layer=act_layer)
        self.drop2=nn.Dropout(0.2)
    def forward(self,x):
        x=x+self.drop1(self.atten(self.norm1(x)))
        x=x+self.drop2(self.mlp(self.norm2(x)))
        return x
class Vit(nn.Module):
    def __init__(self,in_channels,embed_dim,img_size_h,img_size_w,kernel_size,num_heads,mlp_ratio,depth,stride,act_layer=nn.GELU):
        super(Vit,self).__init__()
        self.patch_embed=PatchemBed(in_channels=in_channels,embed_dim=embed_dim,img_size_h=img_size_h,img_size_w=img_size_w,kernel_size=kernel_size,stride=stride)
        num_patches=self.patch_embed.num_patches
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches,embed_dim))
        self.blocks=nn.Sequential(*[
            Block(embed_dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        x=self.patch_embed(x)
        x=x + self.pos_embed
        x = self.blocks(x)
        x=self.norm(x)
        return x
'''block=Vit(embed_dim=768,num_heads=4,mlp_ratio=2,act_layer=nn.GELU,in_channels=3,kernel_size=16,depth=8,img_size_h=256,img_size_w=512)
x=block(img).transpose(1,2).reshape(1,3,256,512)
print(x.shape)'''
