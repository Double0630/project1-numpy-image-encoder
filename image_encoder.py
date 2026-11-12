"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        batch, tokens, width = x.shape
        if width != self.width or width % self.heads != 0:
            raise ValueError('feature width must be divisible by the number of heads')
        # [B,N,D] -> [B,N,H,D/H] -> [B,H,N,D/H]
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # Compare every query token with every key token in each head.
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # [B,H,N,N] @ [B,H,N,D/H] -> [B,H,N,D/H], then restore token order.
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.width)

        # Stored linear weights are [out,in].
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        batch, channels, height, width = images.shape
        patch_size = 32
        if channels != 3 or height != 224 or width != 224:
            raise ValueError('expected input shape [B,3,224,224]')
        rows = height // patch_size
        columns = width // patch_size
        # [B,C,rows,patch,cols,patch] -> [B,rows,cols,C,patch,patch]
        patches = images.reshape(batch, channels, rows, patch_size, columns, patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, rows * columns, channels * patch_size * patch_size)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        tokens = np.concatenate((cls, projected), axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Keep a singleton token axis so post-Norm applies to the CLS row.
        cls = tokens[:, 0, :]
        normalized = self.post_norm(cls[:, None, :])[:, 0, :]
        output = normalized @ self.projection

        return unit_vector(output)
