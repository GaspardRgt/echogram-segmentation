from torch import nn
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class Encoder(nn.Module):
    def __init__(self, channels=[4, 16, 32, 64]):
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []

        for block in self.encBlocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        
        return block_outputs
    
class Decoder(nn.Module):
    def __init__(self, channels=[64, 32, 16]):
        super().__init__()
        # initialize the number of channels, upsampler blocks and decoder blocks
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=channels[i],
                                out_channels=channels[i + 1],
                                kernel_size=2,
                                stride=2)
            for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.batchnorm = nn.BatchNorm2d(num_features=channels[-1])

    def crop(self, enc_features, x):
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)

        return enc_features
    
    def forward(self, x, enc_features):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks, concatenate them with the current upsampled features, pass trough the decoder block
            encFeat = self.crop(enc_features[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)

        return self.batchnorm(x)
    
class UNet(nn.Module):
    def __init__(self,
                 enc_channels=[4, 16, 32, 64],
                 dec_channels=[64, 32, 16],
                 nb_classes=9,
                 retain_dim=True,
                 out_size=(224, 224)):
        super().__init__()

        # initialize encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(in_channels=dec_channels[-1],
                              out_channels=nb_classes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.retain_dim = retain_dim
        self.out_size = out_size
    
    def forward(self, x):
        # grab the features from the encoder and reverse the order
        enc_features = self.encoder(x)[::-1]

        # pass the encoder features through decoder making sure that their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[0], enc_features[1:])

        # pass the decoder features through the regression head to obtain the segmentation mask
        map = self.head(dec_features)

        # check to see if we are retaining the original output dimensions and if so, resize
        if self.retain_dim:
            map = F.interpolate(map, self.out_size)
        
        return map