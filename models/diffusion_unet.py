import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, cond_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )
            
        if cond_dim is not None:
            self.cond_conv = nn.Conv1d(cond_dim, out_channels * 2, 1)

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        
        scale, shift = 1, 0
        
        # FiLM-like conditioning from time embedding
        if time_emb is not None:
            time_emb = self.time_mlp(time_emb) # (B, 2*C)
            time_emb = time_emb.unsqueeze(-1) # (B, 2*C, 1)
            scale_t, shift_t = time_emb.chunk(2, dim=1)
            scale = scale * (scale_t + 1)
            shift = shift + shift_t
            
        # Conditioning from external feature (e.g. IMU features)
        if cond is not None:
            # Assume cond is already up/down-sampled to match x's length if needed
            # Or use global pooling if cond is global. Here we assume sequential cond.
            cond_feat = self.cond_conv(cond) # (B, 2*C, L)
            scale_c, shift_c = cond_feat.chunk(2, dim=1)
            scale = scale * (scale_c + 1)
            shift = shift + shift_c
            
        h = h * scale + shift
        h = self.block2(h)
        
        return h + self.shortcut(x)

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class DiffUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, base_channels=64, 
                 channel_mults=(1, 2, 4, 8), layers_per_block=2):
        super().__init__()
        
        self.channels = base_channels
        self.in_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        
        # Time Embedding
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Condition (IMU) projection
        # We project the high-dim IMU features to match intermediate channels if needed,
        # but ResidualBlock handles projection internally.
        # However, we might want a global adapter.
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        curr_channels = base_channels
        
        # Downsampling
        skip_channels = [curr_channels]
        
        for i, mult in enumerate(channel_mults):
            out_channels_i = base_channels * mult
            for _ in range(layers_per_block):
                self.downs.append(ResidualBlock1D(curr_channels, out_channels_i, time_dim, cond_channels))
                curr_channels = out_channels_i
                skip_channels.append(curr_channels)
            
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample1D(curr_channels))
                skip_channels.append(curr_channels)
                
        # Middle
        self.mid_block1 = ResidualBlock1D(curr_channels, curr_channels, time_dim, cond_channels)
        self.mid_block2 = ResidualBlock1D(curr_channels, curr_channels, time_dim, cond_channels)
        
        # Upsampling
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels_i = base_channels * mult
            
            if i < len(channel_mults) - 1:
                self.ups.append(Upsample1D(curr_channels))
                # Skip connection channel check
                # Upsample doesn't change channels, but it consumes one skip from Downsample
                
            for _ in range(layers_per_block + 1): # +1 for the layer after upsample or initial concat
                skip = skip_channels.pop()
                self.ups.append(ResidualBlock1D(curr_channels + skip, out_channels_i, time_dim, cond_channels))
                curr_channels = out_channels_i
                
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, curr_channels),
            nn.SiLU(),
            nn.Conv1d(curr_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, cond=None):
        """
        x: (B, in_channels, L) - Noisy velocity
        timesteps: (B,)
        cond: (B, cond_channels, L) - IMU feature map. 
              Must align with x in length, or be handled by blocks.
              Since this is 1D UNet, we assume L is consistent or we pool cond.
              Here we assume cond has same Length as x, and we downsample it manually or inside blocks.
              Wait, if ResBlocks are at different scales, cond needs to scale too.
              
              Strategy: Downsample cond alongside x or rely on broadcasting/interpolation.
              Simplest: Interpolate cond to current resolution `h.shape[-1]` inside each block or before passing.
        """
        t = self.time_mlp(timesteps)
        
        h = self.in_conv(x)
        skips = [h]
        
        # Down
        for layer in self.downs:
            if isinstance(layer, Downsample1D):
                h = layer(h)
                skips.append(h)
            else:
                # Resize cond to current resolution
                curr_cond = F.interpolate(cond, size=h.shape[-1], mode='linear', align_corners=False) if cond is not None else None
                h = layer(h, t, curr_cond)
                skips.append(h)
                
        # Mid
        curr_cond = F.interpolate(cond, size=h.shape[-1], mode='linear', align_corners=False) if cond is not None else None
        h = self.mid_block1(h, t, curr_cond)
        h = self.mid_block2(h, t, curr_cond)
        
        # Up
        for layer in self.ups:
            if isinstance(layer, Upsample1D):
                h = layer(h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                curr_cond = F.interpolate(cond, size=h.shape[-1], mode='linear', align_corners=False) if cond is not None else None
                h = layer(h, t, curr_cond)
                
        return self.out_conv(h)
