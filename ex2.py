import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

# =========== 设备 ===========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========== 网络定义（不变） ===========
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

# =========== 辅助：固定路径生成（仅用于初始化提案） ===========
def generate_brownian_motion_fixed(T, N, d, M, device, seed=1234):
    torch.manual_seed(seed); np.random.seed(seed)
    dt = T / N
    t_grid = torch.linspace(0, T, N + 1, device=device)
    dW = torch.randn(M, N, d, device=device) * np.sqrt(dt)
    return t_grid, dW, dt

# =========== 例子2（原“Example 6”）的前向与解析解 ===========
def forward_process_example3(x0, t_grid, dW, N, d, M):
    X = torch.zeros(M, N + 1, d, device=x0.device)
    X[:, 0, :] = x0
    sigma = 1.0 / np.sqrt(d)
    for i in range(N):
        X[:, i + 1, :] = X[:, i, :] + sigma * dW[:, i, :]
    return X

def analytic_solution_example3(t, X, T):
    # 规范 t 到 (batch,1)
    if isinstance(t, (float, int)):
        t = torch.full((X.shape[0], 1), float(t), device=X.device)
    elif t.dim() == 0:
        t = torch.full((X.shape[0], 1), float(t.item()), device=X.device)
    elif t.dim() == 1:
        t = t.unsqueeze(1)
    d = X.shape[1]
    term_AB = ((T - t) / float(d)) * ((torch.sin(X) * (X < 0).float()).sum(dim=1, keepdim=True)
                                      + (X * (X >= 0).float()).sum(dim=1, keepdim=True))
    idx = torch.arange(1, d + 1, device=X.device).float().unsqueeze(0)
    cos_arg = (idx * X).sum(dim=1, keepdim=True)
    return term_AB + torch.cos(cos_arg)

# =========== A(X), B(X), C(d) ===========
def A_of_X(X):
    d = X.shape[1]
    return (torch.sin(X) * (X < 0).float()).sum(dim=1, keepdim=True) / float(d)
def B_of_X(X):
    d = X.shape[1]
    return (X * (X >= 0).float()).sum(dim=1, keepdim=True) / float(d)
def C_of_d(d):
    return (d + 1) * (2 * d + 1) / 12.0

# =========== 关键新增：可训练随机样本（LRV） ===========
class LearnableSamples(nn.Module):
    """
    存标准正态参数 theta_xi（可训练），前向还原 dW = sqrt(dt)*theta_xi，并由此重建轨迹。
    """
    def __init__(self, M, N, d, dt, init_xi=None):
        super().__init__()
        if init_xi is None:
            xi = torch.randn(M, N, d, device=device)  # ~ N(0,1)
        else:
            xi = init_xi.clone().to(device)
        self.theta_xi = nn.Parameter(xi)   # 可训练
        self.sqrt_dt = float(np.sqrt(dt))
    def dW(self):
        return self.theta_xi * self.sqrt_dt
    def rebuild(self, x0, t_grid):
        dW = self.dW()
        M, N, d = dW.shape
        X = forward_process_example3(x0, t_grid, dW, N, d, M)
        # 注意：这里返回的是 (X, dW) —— 只有两个值
        return X, dW

# =========== 兼容性辅助（新增） ===========
def _safe_rebuild_first3(rebuild_out):
    """兼容 theta.rebuild 返回 1/2/3 个值，统一拿到 (X, Y|None, Z|None)。"""
    if isinstance(rebuild_out, (tuple, list)):
        x = rebuild_out[0] if len(rebuild_out) >= 1 else None
        y = rebuild_out[1] if len(rebuild_out) >= 2 else None
        z = rebuild_out[2] if len(rebuild_out) >= 3 else None
        return x, y, z
    else:
        return rebuild_out, None, None

# =========== 训练（SDMC + LRV，损失保持你们的写法） ===========
def train_sdmc_strict_example3(config, t_grid_fixed=None, dW_fixed=None, run_seed=None):
    # 1) 种子
    seed = run_seed if run_seed is not None else config["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    # 2) 取配置
    d = config["d"]; T = config["T"]; N = config["N"]; M = config["M"]
    hidden_layers = config["hidden_layers"]; hidden_dim_offset = config["hidden_dim_offset"]
    steps_per_time = config["steps_per_time"]; K_alt = config["K_alt"]; batch_size = config["batch_size"]
    lr = config["lr"]; weight_decay = config["weight_decay"]

    dt = T / N
    input_dim = 1 + d
    hidden_dim = hidden_dim_offset + d

    # 3) 时间网格（固定或新建）
    if t_grid_fixed is not None:
        t_grid = t_grid_fixed
    else:
        t_grid = torch.linspace(0, T, N + 1, device=device)

    # 4) LRV 初始化（从固定 dW 反解出标准正态 xi 作为初值）
    if dW_fixed is None:
        _, dW_init, _ = generate_brownian_motion_fixed(T, N, d, M, device, seed)
    else:
        dW_init = dW_fixed
    xi_init = dW_init / np.sqrt(dt)
    theta = LearnableSamples(M, N, d, dt, init_xi=xi_init).to(device)

    # 5) 网络
    models_Y = [FBSDE_Y_Net(input_dim, hidden_layers, hidden_dim).to(device) for _ in range(N)]
    models_Z = [FBSDE_Z_Net(input_dim, d, hidden_layers, hidden_dim).to(device) for _ in range(N)]

    # 6) 统一 Adam（包含 Θ + 所有 Y/Z）
    params = list(theta.parameters())
    for m in models_Y + models_Z:
        params += list(m.parameters())
    opt = Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    # 如需学习率衰减，可换 MultiStepLR；默认不衰减：
    # sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15000, 25000], gamma=0.1)

    # 7) 初始状态（确定 x0）
    x0_val = float(config.get("x0", 0.0))
    x0 = torch.full((M, d), x0_val, device=device)

    # 8) 先用当前 Θ 重建一次路径与 Y_T
    def rebuild_all():
        X, dW = theta.rebuild(x0, t_grid)   # 注意：这里返回 2 个值
        X_T = X[:, -1, :]
        Y_T = analytic_solution_example3(torch.tensor(T, device=device), X_T, T).detach()
        return X, dW, Y_T

    X, dW_cur, Y_T = rebuild_all()
    Y_next = Y_T.clone()

    # 9) 反向时间推进
    for i in reversed(range(N)):
        # 每个时间步开始前，用当前 Θ 重建，保证一致性
        with torch.no_grad():
            X, dW_cur, Y_T = rebuild_all()

        t_i_full = torch.full((M,), float(t_grid[i].item()), device=device)
        X_i_full = X[:, i, :]
        dW_i_full = dW_cur[:, i, :]

        S_Z = steps_per_time // 2
        S_Y = steps_per_time - S_Z

        for alt in range(K_alt):
            # === 训练 Z ===
            loss_Z_epoch = 0.0
            for step in range(S_Z):
                idx = torch.randint(0, M, (min(batch_size, M),), device=device)
                t_b = t_i_full[idx]; X_b = X_i_full[idx, :]; dW_b = dW_i_full[idx, :]
                Ynext_b = Y_next[idx, :].detach()

                opt.zero_grad()
                Z_pred = models_Z[i](t_b, X_b)
                Z_target = (Ynext_b * dW_b) / dt
                loss_Z = torch.mean((Z_pred - Z_target) ** 2)
                loss_Z.backward()
                opt.step()
                loss_Z_epoch += loss_Z.item()
            loss_Z_epoch /= max(1, S_Z)

            # === 训练 Y ===
            loss_Y_epoch = 0.0
            for step in range(S_Y):
                idx = torch.randint(0, M, (min(batch_size, M),), device=device)
                t_b = t_i_full[idx]; X_b = X_i_full[idx, :]; Ynext_b = Y_next[idx, :].detach()

                opt.zero_grad()
                with torch.no_grad():
                    Z_fixed = models_Z[i](t_b, X_b)  # 保持原接口；f 不用 Z 也可
                Y_pred = models_Y[i](t_b, X_b)

                A_val = A_of_X(X_b)
                B_val = B_of_X(X_b)
                C_val = C_of_d(d)
                idx_ar = torch.arange(1, d + 1, device=device).float().unsqueeze(0)
                cos_arg = (idx_ar * X_b).sum(dim=1, keepdim=True)

                f_val = (1.0 + (T - t_b).unsqueeze(1) / (2.0 * d)) * A_val + B_val + C_val * torch.cos(cos_arg)

                residual = Y_pred - (Ynext_b + (T / N) * f_val)
                loss_Y = torch.mean(residual ** 2)
                loss_Y.backward()
                opt.step()
                loss_Y_epoch += loss_Y.item()
            loss_Y_epoch /= max(1, S_Y)

        # 回代 Y_next
        with torch.no_grad():
            Y_next = models_Y[i](t_i_full, X_i_full).detach()

        # 可选日志
        print(f"Time step {i+1}/{N}: avg loss_Y={loss_Y_epoch:.6e}, avg loss_Z={loss_Z_epoch:.6e}")
        # 若启用学习率调度，这里 step
        # sched.step()

    # 10) 评估 Y0 —— ⚠️ 这里改成“安全解包”，不再假设返回 3 个值
    with torch.no_grad():
        rebuild_out = theta.rebuild(x0, t_grid)            # 可能返回 1/2/3 个
        X_eval, _, _ = _safe_rebuild_first3(rebuild_out)   # 统一拿到 X_eval
        assert X_eval is not None, "theta.rebuild must return X as first output"
        t0 = torch.zeros(M, device=device)
        X0 = X_eval[:, 0, :]
        Y0_num = models_Y[0](t0, X0).mean().item()
    return Y0_num

# =========== 主程序（本地单文件自测用，可保留/可移除） ===========
if __name__ == "__main__":
    config = {
        "d": 10, "T": 1.0, "N": 120, "M": 10000,
        "hidden_layers": 2, "hidden_dim_offset": 20,
        "steps_per_time": 3000, "K_alt": 2, "batch_size": 512,
        "lr": 1e-3, "weight_decay": 1e-5, "n_runs": 10, "seed": 2,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "x0": 0.5
    }

    # 用同一条固定路径作为 Θ 的初始化（提案）
    t_grid_fixed, dW_fixed, dt = generate_brownian_motion_fixed(
        config["T"], config["N"], config["d"], config["M"], device, config["seed"]
    )

    Y0_list = []
    for run in range(config["n_runs"]):
        print(f"\n=== Run {run+1}/{config['n_runs']} ===")
        Y0_num = train_sdmc_strict_example3(
            config, t_grid_fixed=t_grid_fixed, dW_fixed=dW_fixed, run_seed=config["seed"] + run
        )
        print(f"Run {run+1}: Y0_num = {Y0_num:.6f}")
        Y0_list.append(Y0_num)

    Y0_array = np.array(Y0_list)
    avg_Y0 = np.mean(Y0_array); std_Y0 = np.std(Y0_array)

    # 理论值（t=0, X0=常量向量）
    x0_val = float(config.get("x0", 0.0))
    X0_single = torch.full((1, config["d"]), x0_val, device=device)
    with torch.no_grad():
        Y0_true_tensor = analytic_solution_example3(torch.tensor(0.0, device=device), X0_single, config["T"])
        Y0_true = float(Y0_true_tensor.squeeze().cpu().numpy())
    rel_err = abs(avg_Y0 - Y0_true) / (abs(Y0_true) + 1e-16) * 100.0

    print("\n=== Summary ===")
    print(f"Using fixed x0 = {x0_val} for each component.")
    print(f"Theoretical Y0 (analytic) : {Y0_true:.6f}")
    print(f"Averaged value             : {avg_Y0:.6f}")
    print(f"Standard deviation         : {std_Y0:.6f}")
    print(f"Relative error (%)         : {rel_err:.6f}")
