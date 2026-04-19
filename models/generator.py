import torch
import torch.nn as nn
import functools

class UNetGenerator(nn.Module):
    """UNet-256 generator as described in paper §III-A.
    
    Uses BatchNorm2d (not InstanceNorm) and skip-connections via concatenation.
    Dropout(0.5) applied to the 3 innermost decoder layers.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64, num_downs: int = 8):
        """Construct a UNet-256 generator.
        
        Args:
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output image channels.
            ngf (int): Number of filters in the first conv layer.
            num_downs (int): Number of downsampling layers. 8 for 256×256.
        """
        super(UNetGenerator, self).__init__()
        
        norm_layer = nn.BatchNorm2d  # [paper §III-A]: BatchNorm2d, NOT InstanceNorm2d
        
        # Build UNet from inside out
        # Innermost block (bottleneck)
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True
        )
        # 3 intermediate blocks with dropout (innermost 3 decoder layers)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=True  # Dropout on 3 innermost
            )
        # Gradually reduce channels
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # Outermost block
        self.model = UnetSkipConnectionBlock(
            out_channels, ngf, input_nc=in_channels, submodule=unet_block,
            outermost=True, norm_layer=norm_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines one UNet submodule (encoder block + decoder block + skip connection)."""
    
    def __init__(self, outer_nc: int, inner_nc: int, input_nc: int = None,
                 submodule=None, outermost: bool = False, innermost: bool = False,
                 norm_layer=nn.BatchNorm2d, use_dropout: bool = False):
        """
        Args:
            outer_nc: Number of output channels of this block.
            inner_nc: Number of channels in the inner conv layer.
            input_nc: Number of input channels (defaults to outer_nc).
            submodule: Sub-block nested inside this block.
            outermost: If True, this is the outermost block.
            innermost: If True, this is the bottleneck block.
            norm_layer: Normalization layer class (BatchNorm2d).
            use_dropout: If True, insert Dropout(0.5) in decoder.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        # BatchNorm2d does not use bias
        use_bias = False
            
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection (except outermost)."""
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
