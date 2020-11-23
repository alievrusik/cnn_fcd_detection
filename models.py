import torch
from torch import nn

class ConvolutionBlock(nn.Module):
    def __init__(self, in_c, out_c, pad=1, p=0., max_pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=pad, bias=True)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p)
        self.maxpool = nn.MaxPool2d(2) if max_pool else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        return x

class DeConvolutionBlock(nn.Module):
    def __init__(self, in_c, out_c, output_pad=0, pad=0, stride=1, p=0., use_relu=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, output_padding=output_pad, padding=pad, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_c)
        if use_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()
        self.dropout = nn.Dropout2d(p)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


def get_conv_branch(p):
    return nn.Sequential(
            ConvolutionBlock(2, 16),
            ConvolutionBlock(16, 32, p=p),
            ConvolutionBlock(32, 64, max_pool=True),
            ConvolutionBlock(64, 128, p=p, max_pool=True),
            ConvolutionBlock(128, 256, p=p, max_pool=True),
        )

def get_deconv_branch(p):
    return nn.Sequential(
            DeConvolutionBlock(256, 128, stride=2, output_pad=1, pad=1),
            DeConvolutionBlock(128, 64, stride=2, output_pad=1, pad=1),
            DeConvolutionBlock(64, 32, stride=2, output_pad=1, pad=1),
            DeConvolutionBlock(32, 16, pad=1),
            DeConvolutionBlock(16, 2, use_relu=False, pad=1)
        )

class Encoder(nn.Module):
    def __init__(self, h, w, nb_of_dims, latent_dim, p=.4):
        super().__init__()
        self.nb_of_dims = nb_of_dims
        self.conv_blocks = nn.ModuleList()
        for _ in range(nb_of_dims):
            self.conv_blocks.append(get_conv_branch(p=p))

        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(nb_of_dims * h//8 * w//8 * 256, latent_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_after_conv = []
        for i in range(self.nb_of_dims):
            x_after_conv.append(self.conv_blocks[i](x[:, i*2: 2 + i*2]))
        x = torch.cat(x_after_conv, dim=1)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        return x    

class Decoder(nn.Module):
    def __init__(self, h, w, nb_of_dims, latent_dim, p=.4):
        super().__init__()  
        self.fc1 = nn.Linear(latent_dim, nb_of_dims * h//8 * w//8 * 256, bias=True)
        self.bn1 = nn.BatchNorm1d(nb_of_dims * h//8 * w//8 * 256)
        self.nb_of_dims = nb_of_dims
        self.h = h
        self.w = w
        self.deconv_blocks = nn.ModuleList()
        for _ in range(nb_of_dims):
            self.deconv_blocks.append(get_deconv_branch(p=p))


    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x_after_deconv = []
        for i in range(self.nb_of_dims):
            x_slice = x[:, i * self.h//8 * self.w//8 * 256: (i+1) * self.h//8 * self.w//8 * 256]
            x_slice = x_slice.view(-1, 256, self.w//8, self.h//8)
            x_after_deconv.append(self.deconv_blocks[i](x_slice))
        out = torch.cat(x_after_deconv, dim=1)
        return out
       

class PatchAutoEncoder(nn.Module):

    def __init__(self, h, w, nb_of_dims, latent_dim, p):
        super().__init__()
        self.encoder = Encoder(h, w, nb_of_dims, latent_dim, p=p)
        self.decoder = Decoder(h, w, nb_of_dims, latent_dim, p=p)
    
    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out

class PatchModel(nn.Module):

    def __init__(self, h, w, nb_of_dims, latent_dim, p):
        super().__init__()
        self.encoder = Encoder(h, w, nb_of_dims, latent_dim, p=p)
        self.fc1 = nn.Linear(latent_dim, latent_dim//2)
        self.dropout = nn.Dropout(p)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim//2, 1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x