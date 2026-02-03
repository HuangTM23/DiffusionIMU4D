import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler

class DiffusionSystem(nn.Module):
    def __init__(self, encoder, unet, mode="end2end", scheduler=None, prediction_type="epsilon"):
        """
        Args:
            encoder: ResNet1D, outputs features (B, D, L_small)
            unet: DiffUNet1D, inputs (x, t, cond) -> noise
            mode: "end2end" or "residual"
            scheduler: diffusers scheduler instance
            prediction_type: "epsilon" or "sample" (for DDPMScheduler)
        """
        super().__init__()
        self.encoder = encoder
        self.unet = unet
        self.mode = mode
        
        if scheduler is None:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type=prediction_type,
                clip_sample=False 
            )
        else:
            self.scheduler = scheduler
            
        if self.mode == "residual":
            # 简单的投影头，用于从 Encoder 特征得到 v_prior
            # 假设 Encoder 输出 512 通道，目标 3 通道
            # 注意：这需要与 ResNet1D 的输出维度匹配
            # 这里的 512 是硬编码的，实际应用中最好动态获取
            self.prior_head = nn.Conv1d(512, 3, kernel_size=1) 

    def set_scheduler(self, scheduler_type="ddpm", **kwargs):
        """
        Switch scheduler type (ddpm or ddim).
        """
        if scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=False,
                **kwargs
            )
        elif scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=False,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler type {scheduler_type}")

    def forward(self, imu, gt_vel):
        """
        Training forward pass.
        Args:
            imu: (B, 6, L)
            gt_vel: (B, 3, L)
        Returns:
            loss: scalar tensor
        """
        # 1. Encode IMU condition
        # encoder output: (B, 512, L/32)
        cond_feat = self.encoder(imu)
        
        # 2. Determine target for diffusion
        if self.mode == "residual":
            # 预测 v_prior 并上采样到原始分辨率
            # ResNet 特征通常被下采样了，需要 Upsample
            v_prior_feat = self.prior_head(cond_feat)
            v_prior = torch.nn.functional.interpolate(v_prior_feat, size=gt_vel.shape[-1], mode='linear', align_corners=False)
            
            # 扩散目标是残差
            target_x0 = gt_vel - v_prior
            
            # 可以加入辅助 Loss 监督 v_prior (可选)
            # prior_loss = torch.nn.functional.mse_loss(v_prior, gt_vel)
        else:
            target_x0 = gt_vel
            
        # 3. Sample Noise & Timesteps
        batch_size = target_x0.shape[0]
        noise = torch.randn_like(target_x0)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=target_x0.device).long()
        
        # 4. Add Noise (Forward Process)
        noisy_x = self.scheduler.add_noise(target_x0, noise, timesteps)
        
        # 5. Predict Noise (Reverse Process)
        # [Modified] 如果是 residual 模式，将 v_prior 拼接到输入中
        if self.mode == "residual":
            # v_prior: (B, C, L)
            # noisy_x: (B, C, L)
            unet_input = torch.cat([noisy_x, v_prior], dim=1)
        else:
            unet_input = noisy_x
            
        model_pred = self.unet(unet_input, timesteps, cond_feat)
        
        # 6. Compute Loss
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "sample":
            target = target_x0
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
            
        loss = torch.nn.functional.mse_loss(model_pred, target)
        
        if self.mode == "residual":
            # 加上 Prior Loss，权重设为 1.0 (可调)
            # 为了确保 v_prior 有意义，必须监督它
            prior_loss = torch.nn.functional.mse_loss(v_prior, gt_vel)
            loss = loss + prior_loss
            
        return loss

    @torch.no_grad()
    def sample(self, imu, num_inference_steps=50):
        """
        Sampling (Inference).
        """
        self.eval()
        batch_size = imu.shape[0]
        seq_len = imu.shape[-1]
        
        # 1. Encode
        cond_feat = self.encoder(imu)
        
        # 2. Prepare Prior (if residual)
        if self.mode == "residual":
            v_prior_feat = self.prior_head(cond_feat)
            v_prior = torch.nn.functional.interpolate(v_prior_feat, size=seq_len, mode='linear', align_corners=False)
        
        # 3. Init Noise
        # 动态获取输出通道数
        out_channels = self.unet.out_conv[-1].out_channels
        xt = torch.randn(batch_size, out_channels, seq_len, device=imu.device)
        
        # 4. Scheduler Init
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 5. Denoising Loop
        for t in self.scheduler.timesteps:
            # Predict noise/sample
            # timesteps tensor 需要扩展 batch 维度
            t_batch = torch.full((batch_size,), t, device=imu.device, dtype=torch.long)
            
            # [Modified] Residual 模式拼接 v_prior
            if self.mode == "residual":
                unet_input = torch.cat([xt, v_prior], dim=1)
            else:
                unet_input = xt
            
            model_output = self.unet(unet_input, t_batch, cond_feat)
            
            # Step
            # diffusers step return: (prev_sample, pred_original_sample, ...)
            xt = self.scheduler.step(model_output, t, xt).prev_sample
            
        # 6. Final Result
        if self.mode == "residual":
            return xt + v_prior
        else:
            return xt
