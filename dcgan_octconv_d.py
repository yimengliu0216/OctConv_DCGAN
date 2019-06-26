
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import time

os.makedirs("images_d", exist_ok=True)
os.makedirs("models_d", exist_ok=True)
os.makedirs("time", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

#cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def weights_init_normal(m):
    classname = m.__class__.__name__
    
    if classname.find("OctaveConv") != -1:
        if m.conv_l2l is not None:
            torch.nn.init.normal_(m.conv_l2l.weight.data, 0.0, 0.02)
        if m.conv_h2l is not None:
            torch.nn.init.normal_(m.conv_h2l.weight.data, 0.0, 0.02)
        if m.conv_l2h is not None:
            torch.nn.init.normal_(m.conv_l2h.weight.data, 0.0, 0.02)
        if m.conv_h2h is not None:
            torch.nn.init.normal_(m.conv_h2h.weight.data, 0.0, 0.02)
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        if x_h is not None:
            x_h = self.downsample(x_h) if self.stride == 2 else x_h
            x_h2h = self.conv_h2h(x_h)
            x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 else None
        if x_l is not None:
            x_l2h = self.conv_l2h(x_l)
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None 
            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l
    
class dual_channel_upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(dual_channel_upsample, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=scale_factor)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.upsampler(x_h)
        x_l = self.upsampler(x_l)
        return x_h, x_l
    
class dual_channel_batchnorm2d(nn.Module):
    def __init__(self, size, eps, alpha=0.5):
        super(dual_channel_batchnorm2d, self).__init__()
        self.batcher_h = nn.BatchNorm2d(size - int(size*alpha), eps)
        self.batcher_l = nn.BatchNorm2d(int(size*alpha), eps)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher_h(x_h)
        if x_l is not None:
            x_l = self.batcher_l(x_l)
        return x_h, x_l
    
class dual_channel_leakyrelu(nn.Module):
    def __init__(self, factor, inplace=True):
        super(dual_channel_leakyrelu, self).__init__()
        self.batcher = nn.LeakyReLU(factor)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher(x_h)
        if x_l is not None:
            x_l = self.batcher(x_l)
        return x_h, x_l
    
class high_channel_tanh(nn.Module):
    def __init__(self):
        super(high_channel_tanh, self).__init__()
        self.batcher = nn.Tanh()
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher(x_h)
        return x_h

class dual_channel_dropout(nn.Module):
    def __init__(self, p, inplace=False):
        super(dual_channel_dropout, self).__init__()
        self.batcher = nn.Dropout2d(p)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher(x_h)
        if x_l is not None:
            x_l = self.batcher(x_l) 
        return x_h, x_l

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, alpha_in=0.5, alpha_out = 0.5, bn=True):
            block = [OctaveConv(in_filters, out_filters, 3, alpha_in, alpha_out, stride=2, padding=1), 
                     dual_channel_leakyrelu(0.2, inplace=True), 
                     dual_channel_dropout(0.25, inplace=False)]
            if bn:
                block.append(dual_channel_batchnorm2d(out_filters, 0.8, alpha_out))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, alpha_in=0, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128, alpha_out=0),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out, _ = out
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("data/cifar10", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.FloatTensor# if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
       
    t_init = time.time()
    
    for i, (imgs, _) in enumerate(dataloader):
        t = time.time()

        # Adversarial ground truths
        valid = Tensor(imgs.shape[0], 1).fill_(1.0).to(device)
        fake = Tensor(imgs.shape[0], 1).fill_(0.0).to(device)

        # Configure input
        real_imgs = imgs.type(Tensor).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        time_elapse = time.time() - t;
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [time: %f ]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), time_elapse)
        )
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images_d/%d.png" % batches_done, nrow=5, normalize=True)
            torch.save(generator.state_dict(), 'models_d/model_G_%d.pth' % batches_done)
            torch.save(discriminator.state_dict(), 'models_d/model_D_%d.pth' % batches_done)
            
    
    # time calculation  
    time_epoch = time.time() - t_init;  
    print(
    "[Epoch %d/%d] [time: %f ]"
            %(epoch, opt.n_epochs, time_epoch)
    )
        
    curr_path = os.path.abspath('.')
    with open('time/dcgan_time_d.txt', 'a') as f:
        f.write(str(time_epoch) + '\n')
    
    
    