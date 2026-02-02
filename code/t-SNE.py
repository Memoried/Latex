import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.lines as mlines

# --- 设置学术风格 ---
# try:
#     # 尝试使用seaborn的学术风格，如果没有安装则回退到默认
#     import seaborn as sns
#     sns.set_theme(style="white", font="serif")
#     sns.set_palette("deep")
# except ImportError:
#     plt.style.use('seaborn-white')
#     plt.rcParams['font.family'] = 'serif'

# ======================
# 学术论文风格设置（宋体·小四）
# ======================
plt.rcParams["font.sans-serif"] = ["SimSun"]   # 宋体
plt.rcParams['pdf.fonttype'] = 42   # TrueType
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12        # 小四
plt.rcParams["axes.titlesize"] = 12   # 标题
plt.rcParams["axes.labelsize"] = 12   # 坐标轴标题
plt.rcParams["xtick.labelsize"] = 12  # x轴刻度
plt.rcParams["ytick.labelsize"] = 12  # y轴刻度
plt.rcParams["legend.fontsize"] = 12  # 图例

# --- 数据模拟函数 ---
def generate_base_model_data(n_normal=500, n_anomaly=60):
    """
    模拟 Base Model 数据：
    正常样本：有致密核心，但也有发散的长尾。
    异常样本：散落在外围，与正常样本的长尾混杂。
    """
    np.random.seed(42) # 固定随机种子保证可复现

    # 1. 正常样本 (Normal)
    # 核心致密区域 (Core)
    core_data = np.random.randn(int(n_normal * 0.8), 2) * 0.7
    # 边缘破碎/长尾区域 (Tails) - 方差更大
    tail_data = np.random.randn(int(n_normal * 0.2), 2) * 2.2
    X_normal = np.vstack([core_data, tail_data])

    # 2. 异常样本 (Anomaly)
    # 分布在距离中心一定范围外的环状区域，与长尾混杂
    angles = np.random.uniform(0, 2 * np.pi, n_anomaly)
    # 半径范围覆盖长尾区域 (例如 1.5 到 3.5)
    radii = np.random.uniform(1.5, 3.5, n_anomaly) + np.random.randn(n_anomaly) * 0.2
    X_anomaly = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # 模拟的超球体边界半径 (切断了长尾)
    boundary_radius = 1.8

    return X_normal, X_anomaly, boundary_radius

def generate_ours_model_data(n_normal=500, n_anomaly=60):
    """
    模拟 Ours 模型数据：
    正常样本：紧凑、连贯的球状分布，长尾被拉回中心。
    异常样本：被推远，有清晰的 Margin。
    """
    np.random.seed(100) # 不同的种子

    # 1. 正常样本 (Normal)
    # 非常紧凑的球状分布 (方差小)
    X_normal = np.random.randn(n_normal, 2) * 0.8

    # 2. 异常样本 (Anomaly)
    # 分布在更远的区域，形成清晰隔离带
    angles = np.random.uniform(0, 2 * np.pi, n_anomaly)
    # 半径明显大于正常样本区域 (例如 > 3.0)
    radii = np.random.uniform(3.5, 5.0, n_anomaly)
    X_anomaly = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # 模拟的超球体边界半径 (完美包围正常样本)
    boundary_radius = 2.5

    return X_normal, X_anomaly, boundary_radius

# --- 绘图辅助函数 ---
def plot_tsne_subplot(ax, X_norm, X_anom, radius, title, subtitle):
    # 绘制散点
    ax.scatter(X_norm[:, 0], X_norm[:, 1], c='#1f77b4', marker='o',
               edgecolors='k', s=40, alpha=0.8, label='正常样本 (Normal)', zorder=2)
    ax.scatter(X_anom[:, 0], X_anom[:, 1], c='#d62728', marker='X',
               s=60, linewidth=1.5, label='异常样本 (Anomaly)', zorder=3)

    # 绘制模拟的超球体边界 (Decision Boundary)
    circle = Circle((0, 0), radius, fill=False, color='black',
                    linestyle='--', linewidth=2.5, zorder=4, alpha=0.8)
    ax.add_patch(circle)

    # 添加边界的注释
    # 找到圆圈右下方的点进行注释
    angle = -np.pi / 4
    txt_x = radius * np.cos(angle)
    txt_y = radius * np.sin(angle)
    
    # 针对不同子图调整注释位置
    xytext_offset = (40, -40) if "Base" in title else (40, -20)
    connection_style = "arc3,rad=0.2" if "Base" in title else "arc3,rad=0"

    ax.annotate('超球体边界\n(Hypersphere Boundary)',
                xy=(txt_x, txt_y), xycoords='data',
                xytext=xytext_offset, textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle=connection_style,
                                color='black', lw=1.5),
                fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    # 标题和坐标轴设置
    ax.set_title(title, fontsize=14, fontweight='bold', pad=22)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', fontsize=11, fontstyle='italic')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    
    # 设置坐标轴范围以保持视觉一致性
    lim = 6.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(False) # t-SNE通常不显示网格
    
    # 移除刻度数字，强调拓扑结构而非具体数值 (常见做法)
    # ax.set_xticks([])
    # ax.set_yticks([])

# --- 主程序 ---
# 生成数据
X_norm_base, X_anom_base, r_base = generate_base_model_data()
X_norm_ours, X_anom_ours, r_ours = generate_ours_model_data()

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 绘制子图 (a) Base Model
plot_tsne_subplot(axes[0], X_norm_base, X_anom_base, r_base,
                  title="(a) Deep SVDD",
                  subtitle="长尾散落，边界切断，与异常混杂")
# 添加 Base Model 特有的注释：长尾
axes[0].annotate('长尾样本被切断\n(Misclassified Tail)', xy=(-2.5, -2.5), xytext=(-5, -5),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 ha='right', fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.7))


# 绘制子图 (b) Ours Model
plot_tsne_subplot(axes[1], X_norm_ours, X_anom_ours, r_ours,
                  title="(b) Ours (IED-Weighted)",
                  subtitle="分布紧凑，长尾拉回，清晰隔离带")

# 添加 Ours Model 特有的注释：隔离带 (Margin)
# 在边界和异常点之间画一个指示
margin_x = r_ours + 0.8 # 指向空白区域
axes[1].annotate('空白隔离带\n(Clear Margin)',
                 xy=(margin_x, 0), xytext=(margin_x + 2, 0),
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color='black', lw=2),
                 ha='left', va='center', fontsize=11, fontweight='bold')


# 添加全局图例
# 创建自定义图例句柄
normal_handle = mlines.Line2D([], [], color='#1f77b4', marker='o', linestyle='None',
                          markersize=10, markeredgecolor='k', label='正常样本 (Normal)')
anomaly_handle = mlines.Line2D([], [], color='#d62728', marker='X', linestyle='None',
                          markersize=10, label='异常样本 (Anomaly)')
boundary_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.5, label='超球体边界')

fig.legend(handles=[normal_handle, anomaly_handle, boundary_handle],
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, fontsize=12, frameon=True, fancybox=True, shadow=True)

# 调整布局
plt.tight_layout()
# 为底部图例留出空间
plt.subplots_adjust(bottom=0.15)

# 添加总标题 (可选)
# fig.suptitle("图 3-x 特征空间 t-SNE 可视化对比 (SMD 数据集)", fontsize=16, y=1.05)

plt.savefig(
    "figure.pdf",
    bbox_inches="tight",
    pad_inches=0.02
)

plt.show()
