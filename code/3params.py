import matplotlib.pyplot as plt
import numpy as np

# ======================
# 学术论文风格设置（宋体·小四）
# ======================
plt.rcParams["font.sans-serif"] = ["SimSun"]   # 宋体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12        # 小四
plt.rcParams["axes.titlesize"] = 12   # 标题
plt.rcParams["axes.labelsize"] = 12   # 坐标轴标题
plt.rcParams["xtick.labelsize"] = 12  # x轴刻度
plt.rcParams["ytick.labelsize"] = 12  # y轴刻度
plt.rcParams["legend.fontsize"] = 12  # 图例

# --- 2. 构造实验数据（用于超参数敏感性分析） ---

# (a) β：呈倒 U 型变化趋势
# β = 0.0：基线模型（F1 ≈ 90）
# β = 1.5：性能最优（F1 ≈ 95.2）
# β > 2.0：引入噪声，性能下降
beta_x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
beta_y = np.array([90.10, 92.65, 94.80, 95.20, 94.60, 93.10, 91.50])

# (b) α：稳定后急剧下降
# α = 0.01：中心更新缓慢
# α = 0.1：最优更新速率
# α > 0.3：更新震荡，模型不稳定
alpha_x = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
alpha_y = np.array([93.20, 94.90, 95.20, 94.50, 91.50, 87.20, 84.10])

# --- 3. 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# === 子图 (a)：β 的敏感性分析 ===
axes[0].plot(
    beta_x, beta_y,
    marker='o', linestyle='-', linewidth=2.5, markersize=8,
    label='F1 值'
)

# 标注性能峰值
max_beta_idx = np.argmax(beta_y)
axes[0].annotate(
    f'性能峰值：{beta_y[max_beta_idx]}%',
    xy=(beta_x[max_beta_idx], beta_y[max_beta_idx]),
    xytext=(0, 10),
    textcoords='offset points',
    ha='center', va='bottom',
    fontweight='bold', fontsize=11
)

# 标注基线模型
axes[0].annotate(
    '基线模型\n（无权重机制）',
    xy=(0, 90.1),
    xytext=(0.6, 90.6),
    arrowprops=dict(arrowstyle='->', color='gray'),
    fontsize=10, color='gray'
)

axes[0].set_title(r'(a) 多样性敏感系数 $\beta$ 的影响')
axes[0].set_xlabel(r'多样性系数 $\beta$')
axes[0].set_ylabel('F1 值（%）')
axes[0].set_ylim(89, 96)

# === 子图 (b)：α 的敏感性分析 ===
axes[1].plot(
    alpha_x, alpha_y,
    marker='s', linestyle='-', linewidth=2.5, markersize=8,
    label='F1 值'
)

# 标注最优点
max_alpha_idx = np.argmax(alpha_y)
axes[1].annotate(
    f'最优更新率：{alpha_y[max_alpha_idx]}%',
    xy=(alpha_x[max_alpha_idx], alpha_y[max_alpha_idx]),
    xytext=(10, 5),
    textcoords='offset points',
    ha='left', va='center',
    fontweight='bold', fontsize=11
)

# 标注不稳定区域
axes[1].annotate(
    '模型更新震荡区',
    xy=(0.5, 87.2),
    xytext=(0.3, 85.5),
    arrowprops=dict(arrowstyle='->', color='gray'),
    fontsize=10, color='gray'
)

axes[1].set_title(r'(b) 特征中心更新率 $\alpha$ 的影响')
axes[1].set_xlabel(r'中心更新率 $\alpha$')
axes[1].set_ylabel('F1 值（%）')
axes[1].set_ylim(83, 96)

plt.tight_layout()
plt.show()
