import torch
import torch.nn as nn
import torchvision.models as models
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
    
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class StochasticDepth(nn.Module):
    def __init__(self, p, mode="row"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1 - self.p

        if self.mode == "row":
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            raise ValueError("ConvNeXt only supports row mode")

        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor

    def extra_repr(self):
        return f"p={self.p}, mode={self.mode}"


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        drop_prob,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            Permute(0, 3, 1, 2),
        )

        # ✅ 关键：名字必须叫 layer_scale
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

        self.stochastic_depth = StochasticDepth(drop_prob, mode="row")

    def forward(self, x):
        out = self.block(x)
        out = self.layer_scale.view(1, -1, 1, 1) * out
        out = self.stochastic_depth(out)
        return x + out



class ConvNeXtBackbone1(nn.Module):
    def __init__(self, drop_path_rate=0.1):
        super().__init__()

        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        backbone = []

        # Stem
        backbone.append(
            nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                LayerNorm2d(96, eps=1e-6),
            )
        )

        # Stage 1
        backbone.append(
            nn.Sequential(*[
                CNBlock(dims[0], dp_rates[cur + i])
                for i in range(depths[0])
            ])
        )
        cur += depths[0]

        # Down + Stage 2
        backbone.append(nn.Sequential(LayerNorm2d(96), nn.Conv2d(96, 192, 2, 2)))
        backbone.append(
            nn.Sequential(*[
                CNBlock(dims[1], dp_rates[cur + i])
                for i in range(depths[1])
            ])
        )
        cur += depths[1]

        # Down + Stage 3
        backbone.append(nn.Sequential(LayerNorm2d(192), nn.Conv2d(192, 384, 2, 2)))
        backbone.append(
            nn.Sequential(*[
                CNBlock(dims[2], dp_rates[cur + i])
                for i in range(depths[2])
            ])
        )
        cur += depths[2]

        # Down + Stage 4
        backbone.append(nn.Sequential(LayerNorm2d(384), nn.Conv2d(384, 768, 2, 2)))
        backbone.append(
            nn.Sequential(*[
                CNBlock(dims[3], dp_rates[cur + i])
                for i in range(depths[3])
            ])
        )

        self.backbone = nn.Sequential(*backbone)
        # projection head（你现在这个一定是自己加的）
        

    def forward(self, x):
        x = self.backbone(x)
        # print("convnext output shape:", x.shape)
        # print("ConvNeXt1111 output shape after conv:11111", x.shape)
        return {"feature_map": x}
    
if __name__ == "__main__":
    tv_model = models.convnext_tiny(weights='DEFAULT')
    tv_backbone = tv_model.features
    my_backbone = ConvNeXtBackbone1()
    my_backbone.backbone.load_state_dict(tv_backbone.state_dict(),strict=True)
    print(my_backbone.backbone)