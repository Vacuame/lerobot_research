import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. 定义一个非常简单的模型 ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 8)  # y = w*x + b
        self.fc_out = nn.Linear(8, 1)

    def forward(self, x):
        y=self.fc(x)
        return self.fc_out(y)

# --- 2. 准备伪造数据 y = 2*x + 1 ---
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])   # (4,1)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])   # (4,1)

# --- 3. 实例化模型、优化器、损失 ---
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# --- 4. 训练若干步（非常短） ---
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)            # 前向传播
    # print("预测值:", y_pred.squeeze().detach().numpy())
    loss = loss_fn(y_pred, y)    # 计算损失
    # print("损失:", loss.item())
    loss.backward()              # 反向传播，计算梯度
    optimizer.step()             # 更新参数

# 打印训练后参数
for name, param in model.named_parameters():
    print(name, param.data.numpy())

# --- 5. 保存模型参数（推荐方式：state_dict） ---
torch.save(model.state_dict(), "simple_model_state_dict.pth")

# --- 6. 保存 checkpoint（含优化器状态和 epoch，可用于断点续训） ---
checkpoint = {
    'epoch': 100,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}
torch.save(checkpoint, "simple_checkpoint.pth")

# --- 7. 加载示例（新建模型并加载参数） ---
loaded_model = SimpleNet()
loaded_model.load_state_dict(torch.load("simple_model_state_dict.pth"))
loaded_model.eval()

# 推理
with torch.no_grad():
    print("输入 5 -> 预测:", loaded_model(torch.tensor([[5.0]])).item())
