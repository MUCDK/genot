import flax.linen as nn
import jax.numpy as jnp

class Block(nn.Module):
    dim: int = 32
    groups: int = 8

    @nn.compact
    def __call__(self, inputs):
        conv = nn.Conv(self.dim, (3, 3))(inputs)
        norm = nn.GroupNorm(num_groups=self.groups)(conv)
        activation = nn.silu(norm)
        return activation


class ResnetBlock(nn.Module):
    dim: int = 32
    groups: int = 8

    @nn.compact
    def __call__(self, inputs):
        x = Block(self.dim, self.groups)(inputs)
        x = Block(self.dim, self.groups)(x)
        res_conv = nn.Conv(self.dim, (1, 1), padding="SAME")(inputs)
        return res_conv + inputs # in the original code they do res_conv+x, but don't understand why


class UNet(nn.Module): #adapted from https://www.kaggle.com/code/darshan1504/exploring-diffusion-models-with-jax
    dim: int = 8
    dim_scale_factor: tuple = (1, 2, 4, 8)
    num_groups: int = 8


    @nn.compact
    def __call__(self, inputs):
        channels = inputs.shape[-1]
        x = nn.Conv(self.dim // 3 * 2, (7, 7), padding=((3,3), (3,3)))(inputs)

        dims = [self.dim * i for i in self.dim_scale_factor]
        pre_downsampling = []

        # Downsampling phase
        for index, dim in enumerate(dims):
            x = ResnetBlock(dim, self.num_groups)(x)
            x = ResnetBlock(dim, self.num_groups)(x)
            norm = nn.GroupNorm(self.num_groups)(x)
            x = norm + x
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if index != len(dims) - 1:
                x = nn.Conv(dim, (4,4), (2,2))(x)

        # Middle block
        x = ResnetBlock(dims[-1], self.num_groups)(x)
        norm = nn.GroupNorm(self.num_groups)(x)
        x = norm + x 
        x = ResnetBlock(dims[-1], self.num_groups)(x)

        # Upsampling phase
        for index, dim in enumerate(reversed(dims)):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = ResnetBlock(dim, self.num_groups)(x)
            x = ResnetBlock(dim, self.num_groups)(x)
            norm = nn.GroupNorm(self.num_groups)(x)
            x = norm + x
            if index != len(dims) - 1:
                x = nn.ConvTranspose(dim, (4,4), (2,2))(x)


        # Final ResNet block and output convolutional layer
        x = ResnetBlock(dim, self.num_groups)(x)
        x = nn.Conv(channels, (1,1), padding="SAME")(x)
        return x

    