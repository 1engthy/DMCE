import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import json

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 网络结构定义 =====
class FeedForwardUZ(nn.Module):
    def __init__(self, input_dim, hidden_dim=11, hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t, x):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.network(torch.cat([t, x], dim=1))


class FeedForwardGam(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=11, hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, t, x):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.network(torch.cat([t, x], dim=1))


# ===== 解析解 Example 1 =====
def example1_sol(t, x):
    if isinstance(t, float) or isinstance(t, int):
        t = torch.full((x.shape[0], 1), float(t), device=x.device)
    elif t.dim() == 0:
        t = torch.full((x.shape[0], 1), float(t.item()), device=x.device)
    elif t.dim() == 1:
        t = t.unsqueeze(1)
    d = x.shape[1]
    inner = (t + x.sum(dim=1, keepdim=True) / d).clamp(-30, 30)
    return torch.sigmoid(inner)


# ===== 路径生成 =====
def generate_brownian_motion_fixed(T, N, d, M, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    dt = T / N
    t_grid = torch.linspace(0, T, N + 1, device=device)
    dW = torch.randn(M, N, d, device=device) * np.sqrt(dt)
    return t_grid, dW, dt


def forward_process_from_dW(x0, sigma, dW):
    M, N, d = dW.shape
    X = torch.zeros(M, N + 1, d, device=x0.device)
    X[:, 0, :] = x0
    for i in range(N):
        X[:, i + 1, :] = X[:, i, :] + sigma * dW[:, i, :]
    return X


# ===== 主训练过程 =====
def run_example1_dbdp1(seed=None):
    d = 80
    T = 1.0
    N, picard = 120, 30
    M = 10000
    batch_size = 1000
    hidden_dim = 10 + d
    hidden_layers = 2
    sigma = d / np.sqrt(2)
    learning_rate = 1e-3
    outer_loops = 10

    x0 = torch.zeros((M, d), device=device)
    t_grid, dW, dt = generate_brownian_motion_fixed(T, N, d, M, device, seed)
    X = forward_process_from_dW(x0, sigma, dW)

    modelU = FeedForwardUZ(d + 1, hidden_dim, hidden_layers).to(device)
    modelZ = FeedForwardGam(d + 1, d, hidden_dim, hidden_layers).to(device)
    opt = Adam(list(modelU.parameters()) + list(modelZ.parameters()), lr=learning_rate)

    for outer in range(outer_loops):
        for i in range(N):
            t_i = t_grid[i].item()
            t = torch.full((M,), t_i, device=device)
            x = X[:, i, :]
            x_next = X[:, i + 1, :]
            dW_i = dW[:, i, :]
            t_next = torch.full((M,), t_grid[i + 1].item(), device=device)

            y_next = example1_sol(t_next, x_next).detach()

            opt.zero_grad()
            y_pred = modelU(t, x)
            z_pred = modelZ(t, x)
            sumZ = torch.sum(z_pred, dim=1, keepdim=True)
            f = (y_next - (d + 2) / (2 * d)) * (sumZ / sigma)
            target = y_next + dt * f
            loss = torch.mean((y_pred - target) ** 2)
            loss.backward()
            opt.step()

    with torch.no_grad():
        t0 = torch.zeros(M, device=device)
        x0 = X[:, 0, :]
        Y0_pred = modelU(t0, x0).mean().item()
        Y0_true = 0.5
    print("预测值: {:.6f}, 理论解: {:.6f}, 误差: {:.6f}".format(Y0_pred, Y0_true, abs(Y0_pred - Y0_true)))
    return Y0_pred


# ===== 多轮运行并输出 JSON 结构 =====
if __name__ == '__main__':
    results = []
    for i in range(10):
        print(f"\n🔁 第 {i+1} 次运行：")
        y = run_example1_dbdp1(seed=i)
        results.append(y)

    results = np.array(results)
    Y0_true = 0.5
    diffs = np.abs(results - Y0_true)

    output = {
        "params": {
            "d": 80,
            "T": 1.0,
            "N": 120,
            "M": 10000,
            "hidden_layers": 2,
            "hidden_dim_offset": 10,
            "outer_loops": 10,
            "batch_size": 1000,
            "lr": 1e-3,
            "n_runs": 10,
            "seed": "0~9"
        },
        "avg_Y0": float(np.mean(results)),
        "std_Y0": float(np.std(results)),
        "runs": results.tolist(),
        "diffs": diffs.tolist(),
        "max_diff": float(np.max(diffs)),
        "avg_diff": float(np.mean(diffs)),
        "Y0_true": float(Y0_true)
    }

    print("\n=== 输出 JSON 格式结果 ===")
    print(json.dumps(output, indent=4, ensure_ascii=False))

    with open("example1_dbdp1_output_d_80.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
        print("✅ 已保存至 example1_dbdp1_output_d_80.txt")
