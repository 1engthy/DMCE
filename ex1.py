# sd_mce_strict_fixed_config_strictSDMC.py (LRV集成版，保留你们损失与Adam)
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 网络定义（不变） =====
class FBSDE_Y_Net(nn.Module):
    def __init__(self, input_dim, hidden_layers=2, hidden_dim=200):
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

class FBSDE_Z_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_dim=200):
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

# ===== 路径生成（仅用于时间网格和初始化） =====
def generate_brownian_motion_fixed(T, N, d, M, device, seed=1234):
    torch.manual_seed(seed); np.random.seed(seed)
    dt = T / N
    t_grid = torch.linspace(0, T, N + 1, device=device)
    dW = torch.randn(M, N, d, device=device) * np.sqrt(dt)
    return t_grid, dW, dt

# ===== 解析解（不变） =====
def analytic_solution(t, W, d):
    sum_W = torch.sum(W, dim=-1) / d
    exponent = torch.clamp(t + sum_W, -30.0, 30.0)
    return torch.exp(exponent) / (1.0 + torch.exp(exponent))

# ======= 关键新增：把MC样本变成“可训练的随机变量” =======
class LearnableSamples(nn.Module):
    """
    可训练的布朗增量参数 Θ，初始化为标准正态，再按 √dt 缩放得到 dW。
    训练时直接把 Θ 当作参数用 Adam 更新（LRV 思路）。
    """
    def __init__(self, M, N, d, dt, init_xi=None):
        super().__init__()
        if init_xi is None:
            xi = torch.randn(M, N, d, device=device)  # ~ N(0,1)
        else:
            xi = init_xi.clone().to(device)  # 期望 init_xi ~ N(0,1)
        self.theta_xi = nn.Parameter(xi)  # 可训练
        self.sqrt_dt = float(np.sqrt(dt))
    def dW(self):
        return self.theta_xi * self.sqrt_dt  # 还原成时间步尺度的 dW
    def W_path(self):
        return torch.cumsum(self.dW(), dim=1)  # (M,N,d)

def forward_process_from_dW(x0, sigma, dW):
    M, N, d = dW.shape
    X = torch.zeros(M, N + 1, d, device=x0.device)
    X[:, 0, :] = x0
    for i in range(N):
        X[:, i + 1, :] = X[:, i, :] + sigma * dW[:, i, :]
    return X

# =========== 主训练函数（严格 SDMC + LRV） ===========
def train_sdmc_strict(config, t_grid_fixed=None, dW_fixed=None, run_seed=None):
    # 固定种子
    seed = run_seed if run_seed is not None else config["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    d = config["d"]; T = config["T"]; N = config["N"]; M = config["M"]
    hidden_layers = config["hidden_layers"]; hidden_dim_offset = config["hidden_dim_offset"]
    steps_per_time = config["steps_per_time"]; K_alt = config["K_alt"]; batch_size = config["batch_size"]
    lr = config["lr"]; weight_decay = config["weight_decay"]

    dt = T / N
    sigma = d / np.sqrt(2)        # 保留你的设定
    input_dim = 1 + d
    hidden_dim = hidden_dim_offset + d

    # 时间网格（可复用固定的）
    if t_grid_fixed is not None:
        t_grid = t_grid_fixed
    else:
        t_grid = torch.linspace(0, T, N + 1, device=device)

    # ===== LRV 初始化：把固定的 dW 反解成标准正态 xi 作为 Θ 的初值 =====
    if dW_fixed is None:
        _, dW_init, _ = generate_brownian_motion_fixed(T, N, d, M, device, seed)
    else:
        dW_init = dW_fixed
    xi_init = dW_init / np.sqrt(dt)  # 标准正态尺度
    theta = LearnableSamples(M, N, d, dt, init_xi=xi_init).to(device)  # ★ 可训练样本 Θ

    # 每个时间步的 Y/Z 网络（与原 SDMC 相同）
    models_Y = [FBSDE_Y_Net(input_dim, hidden_layers, hidden_dim).to(device) for _ in range(N)]
    models_Z = [FBSDE_Z_Net(input_dim, d, hidden_layers, hidden_dim).to(device) for _ in range(N)]

    # 优化器：统一 Adam —— 同时更新 {Θ} 与 {Y_i,Z_i} 参数（你们要求）
    params = list(theta.parameters())
    for m in models_Y + models_Z:
        params += list(m.parameters())
    opt = Adam(params, lr=lr, weight_decay=weight_decay)
    # （可选）简洁学习率调度：统一调度器，避免对每个时间步单独一个
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.9)

    # 基于 Θ 重建全路径（后面每个时间步都会用）
    def rebuild_states_from_theta():
        dW = theta.dW()                 # (M,N,d)
        X  = forward_process_from_dW(torch.zeros(M, d, device=device), sigma, dW)  # (M,N+1,d)
        Wp = torch.cumsum(dW, dim=1)    # (M,N,d)
        return X, dW, Wp

    # ★ 改1：终端条件应基于 X_T，而不是 W_path
    X, dW_cur, _ = rebuild_states_from_theta()
    Y_T = analytic_solution(torch.tensor(T, device=device), X[:, -1, :], d).unsqueeze(1).detach()

    # 反向时间推进：交替训练 Z 与 Y（保留你们损失设计）
    Y_next = Y_T.clone()
    for i in reversed(range(N)):
        # 由于 Θ 会被更新，进入每个时间步前用当前 Θ 重建状态，保证一致性
        with torch.no_grad():
            X, dW_cur, _ = rebuild_states_from_theta()
            Y_T = analytic_solution(torch.tensor(T, device=device), X[:, -1, :], d).unsqueeze(1)

        t_i_full = torch.full((M,), float(t_grid[i].item()), device=device)
        X_i_full = X[:, i, :]             # (M,d)
        dW_i_full = dW_cur[:, i, :]       # (M,d)

        S_Z = steps_per_time // 2
        S_Y = steps_per_time - S_Z
        loss_Z_epoch = 0.0; loss_Y_epoch = 0.0

        for alt in range(K_alt):
            # ===== 训练 Z（目标仍采用 Y_next），损失定义保持你们原样 =====
            for step in range(S_Z):
                idx = torch.randint(0, M, (min(batch_size, M),), device=device)
                t_b, X_b, dW_b = t_i_full[idx], X_i_full[idx, :], dW_i_full[idx, :]
                Ynext_b = Y_next[idx, :].detach()

                opt.zero_grad()
                Z_pred = models_Z[i](t_b, X_b)
                Z_target = (Ynext_b * dW_b) / dt   # 你们 Example6 的目标构造
                loss_Z = torch.mean((Z_pred - Z_target) ** 2)
                loss_Z.backward()
                opt.step()
                loss_Z_epoch += loss_Z.item()

            # ===== 训练 Y（使用你们的 f 形式），损失定义保持不变 =====
            for step in range(S_Y):
                idx = torch.randint(0, M, (min(batch_size, M),), device=device)
                t_b, X_b = t_i_full[idx], X_i_full[idx, :]
                Ynext_b = Y_next[idx, :].detach()

                opt.zero_grad()
                with torch.no_grad():
                    Z_fixed = models_Z[i](t_b, X_b)               # (batch,d)
                Y_pred = models_Y[i](t_b, X_b)                    # (batch,1)

                sumZ = torch.sum(Z_fixed, dim=1, keepdim=True)  # (batch,1)
                f_val = (Ynext_b - (d + 2) / (2.0 * d)) * (sumZ / sigma)  # ← 用 Ynext_b，且除以 σ

                residual = Y_pred - (Ynext_b + (T/N) * f_val)
                loss_Y = torch.mean(residual ** 2)
                loss_Y.backward()
                opt.step()
                loss_Y_epoch += loss_Y.item()

        # 归一化 epoch 损失
        total_Z_steps = max(1, K_alt * S_Z)
        total_Y_steps = max(1, K_alt * S_Y)
        loss_Z_epoch /= total_Z_steps; loss_Y_epoch /= total_Y_steps

        # 回代更新 Y_next（用当前网络）
        with torch.no_grad():
            Y_next = models_Y[i](t_i_full, X_i_full).detach()

        sched.step()
        print(f"Time step {i+1}/{N}: avg loss_Y = {loss_Y_epoch:.6e}, avg loss_Z = {loss_Z_epoch:.6e}")

    # 评估 Y0
    with torch.no_grad():
        X, _, _ = rebuild_states_from_theta()
        t0 = torch.zeros(M, device=device)
        X0 = X[:, 0, :]
        Y0_num = models_Y[0](t0, X0).mean().item()
    return Y0_num

# =========== 主程序（基本不变） ===========
if __name__ == "__main__":
    config = {
        "d": 100, "T": 1.0, "N": 10, "M": 10000,
        "hidden_layers": 2, "hidden_dim_offset": 110,
        "steps_per_time": 2000, "K_alt": 2, "batch_size": 1000,
        "lr": 1e-3, "weight_decay": 1e-5, "n_runs": 10, "seed": 2
    }

    # 仍生成一份固定路径用于初始化 Θ（proposal），符合 LRV 初始化建议
    t_grid_fixed, dW_fixed, dt = generate_brownian_motion_fixed(
        config["T"], config["N"], config["d"], config["M"], device, config["seed"]
    )

    Y0_list = []
    for run in range(config["n_runs"]):
        print(f"\n=== Run {run+1}/{config['n_runs']} ===")
        Y0_num = train_sdmc_strict(
            config, t_grid_fixed=t_grid_fixed, dW_fixed=dW_fixed, run_seed=config["seed"] + run
        )
        print(f"Run {run+1}: Y0_num = {Y0_num:.6f}")
        Y0_list.append(Y0_num)

    Y0_array = np.array(Y0_list)
    avg_Y0 = np.mean(Y0_array); std_Y0 = np.std(Y0_array)
    rel_err = abs(avg_Y0 - 0.5) / 0.5 * 100.0
    print("\n=== Summary ===")
    print("Theoretical solution: 0.5")
    print(f"Averaged value      : {avg_Y0:.6f}")
    print(f"Standard deviation  : {std_Y0:.6f}")
    print(f"Relative error (%)  : {rel_err:.4f}")
