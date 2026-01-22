
# 直接用本身的还原了，不用自己写的了

# import torch
# def denormalize_img_with_mean_stats(normalized_data):
#         mean = torch.tensor([0.485, 0.456, 0.406], device=normalized_data.device).view(-1, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=normalized_data.device).view(-1, 1, 1)
#         img = normalized_data * std + mean
#         return torch.clamp(img, 0.0, 1.0)

# def denormalize_obs_and_angle_to_rad(
#     q_norm: torch.Tensor,   # (B, N)
#     mean: torch.Tensor,     # (N,)
#     std: torch.Tensor,      # (N,)
# ) -> torch.Tensor:
#     assert q_norm.ndim == 2
#     assert mean.ndim == 1 and std.ndim == 1

#     # 对齐 device / dtype
#     mean = mean.to(q_norm.device, q_norm.dtype)
#     std = std.to(q_norm.device, q_norm.dtype)

#     q_deg = q_norm * std + mean
#     q_rad = q_deg * (torch.pi / 180.0)

#     return q_rad
