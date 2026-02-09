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

    def _compute_integral_loss(self, v_pred, v_gt):
        """
        计算积分位置损失 (Integral/Trajectory Loss)
        通过对速度积分，强制约束轨迹的形状和终点。
        v_pred, v_gt: (B, C, L)
        """
        # 沿着时间维度 (dim=2) 进行累积求和 (Cumsum)
        # 这代表了从 t=0 开始的每一时刻的相对位移轨迹
        # p_pred: (B, C, L), p_gt: (B, C, L)
        p_pred = torch.cumsum(v_pred, dim=2)
        p_gt = torch.cumsum(v_gt, dim=2)
        
        # 计算轨迹误差 (MSE of positions)
        loss = torch.nn.functional.mse_loss(p_pred, p_gt)
        return loss

    def _compute_heading_loss(self, v_pred, v_gt, eps=1e-7):
        """
        计算导航坐标系下的方向损失 (1 - cos_sim)
        v_pred, v_gt: (B, C, L)
        """
        # ... (保留 Heading Loss 作为辅助，或者如果积分损失足够强可以去掉)
        # 为了简洁，这里只保留代码结构，实际调用看需求
        v_pred_norm = torch.norm(v_pred, dim=1, keepdim=True)
        v_gt_norm = torch.norm(v_gt, dim=1, keepdim=True)
        u_pred = v_pred / (v_pred_norm + eps)
        u_gt = v_gt / (v_gt_norm + eps)
        cos_sim = torch.sum(u_pred * u_gt, dim=1)
        motion_mask = (v_gt_norm.squeeze(1) > 0.1).float()
        loss = (1.0 - cos_sim) * motion_mask
        return loss.sum() / (motion_mask.sum() + eps)

    def forward(self, imu, gt_vel):
        """
        Training forward pass.
        """
        # 1. Encode
        cond_feat = self.encoder(imu)
        
        # 2. Target
        if self.mode == "residual":
            v_prior_feat = self.prior_head(cond_feat)
            v_prior = torch.nn.functional.interpolate(v_prior_feat, size=gt_vel.shape[-1], mode='linear', align_corners=False)
            target_x0 = gt_vel - v_prior
        else:
            target_x0 = gt_vel
            v_prior = None
            
        # 3. Noise
        batch_size = target_x0.shape[0]
        noise = torch.randn_like(target_x0)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=target_x0.device).long()
        
        # 4. Forward Process
        noisy_x = self.scheduler.add_noise(target_x0, noise, timesteps)
        
        # 5. Reverse Process
        if self.mode == "residual":
            unet_input = torch.cat([noisy_x, v_prior], dim=1)
        else:
            unet_input = noisy_x
            
        model_pred = self.unet(unet_input, timesteps, cond_feat)
        
        # 6. Base Loss (MSE)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
            total_loss = torch.nn.functional.mse_loss(model_pred, target)
            
            # 如果是 epsilon 模式，由于推导 x0 比较复杂且容易数值不稳定，
            # 暂时只对 Residual 模式下的 Prior 部分加积分约束
            if self.mode == "residual":
                prior_mse = torch.nn.functional.mse_loss(v_prior, gt_vel)
                prior_integral = self._compute_integral_loss(v_prior, gt_vel)
                total_loss = total_loss + prior_mse + 0.5 * prior_integral
            
        elif self.scheduler.config.prediction_type == "sample":
            target = target_x0
            mse_loss = torch.nn.functional.mse_loss(model_pred, target)
            
            # --- Integral Loss ---
            if self.mode == "residual":
                prior_mse = torch.nn.functional.mse_loss(v_prior, gt_vel)
                prior_integral = self._compute_integral_loss(v_prior, gt_vel)
                v_final_pred = model_pred + v_prior
                final_integral = self._compute_integral_loss(v_final_pred, gt_vel)
                total_loss = mse_loss + prior_mse + 0.2 * prior_integral + 0.5 * final_integral
            else:
                final_integral = self._compute_integral_loss(model_pred, gt_vel)
                total_loss = mse_loss + 0.5 * final_integral
                
        else:
            raise ValueError("Unknown prediction type")
            
        return total_loss

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
