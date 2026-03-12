import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# =========================
# 1. 论文级全局参数设置
# =========================
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 小四≈12pt
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 11

# 科研风格线宽
plt.rcParams['axes.linewidth'] = 1.2

# 设置随机种子
np.random.seed(42)

# =========================
# 2. 数据生成（保持不变）
# =========================
n_head = 160
head_data = np.random.normal(loc=0.0, scale=0.35, size=(n_head, 2))

tail_data = np.array([
    [1.6, 1.2], [1.4, -1.3], [-1.7, 1.0],
    [-1.2, -1.6], [1.9, 0.1], [-0.4, 1.8]
])

# =========================
# 3. 画布初始化（高清）
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=600)
plt.subplots_adjust(wspace=0.25)

# 科研配色
deep_red = '#B22222'
deep_blue = '#1f4e79'
deep_green = '#2E8B57'
soft_gray = '#7f7f7f'

# =====================================================
# 左图：传统方法
# =====================================================
ax1 = axes[0]
ax1.set_title("(a) 传统方法（受高频样本主导）", fontweight='bold', pad=15)

ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_aspect('equal')

head_color = "#4C72B0"
tail_color = "#DD8452"

ax1.scatter(head_data[:, 0], head_data[:, 1],
            c=head_color, s=40, alpha=0.65,
            edgecolors='white', linewidth=0.3,
            label='头部样本（高频）')

ax1.scatter(tail_data[:, 0], tail_data[:, 1],
            c=tail_color, s=70, alpha=0.9,
            edgecolors='black', linewidth=0.8,
            label='尾部样本（稀疏）')

# 偏置中心
center1 = (0, 0)
radius1 = 0.85

circle1 = patches.Circle(center1, radius1,
                         linewidth=2.2,
                         edgecolor=deep_red,
                         facecolor='none',
                         linestyle='--')
ax1.add_patch(circle1)

ax1.scatter(*center1,
            c=deep_red,
            marker='x',
            s=150,
            linewidth=3,
            zorder=10,
            label='偏置中心 $c$')

# 误报标注
ax1.annotate('误判为异常\n（精确率下降）',
             xy=tail_data[0],
             xytext=(1.8, 2.1),
             arrowprops=dict(facecolor=deep_red,
                             width=1.2,
                             headwidth=8),
             fontsize=12,
             color=deep_red,
             ha='center',
             fontweight='bold')

ax1.legend(loc='lower left', frameon=True)

# =====================================================
# 右图：多样性加权方法
# =====================================================
ax2 = axes[1]
ax2.set_title("(b) 提出的多样性加权方法", fontweight='bold', pad=15)

ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2.5, 2.5)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')

ax2.scatter(head_data[:, 0], head_data[:, 1],
            c=head_color, s=35, alpha=0.5,
            edgecolors='white', linewidth=0.2)

# 高权重尾部样本
weights = [220] * len(tail_data)

ax2.scatter(tail_data[:, 0], tail_data[:, 1],
            c=tail_color,
            s=weights,
            alpha=0.95,
            edgecolors='black',
            linewidth=1.2,
            label='加权样本（高IED权重）')

# 校正中心
center2 = (0.2, 0.15)
radius2 = 2.15

circle2 = patches.Circle(center2, radius2,
                         linewidth=2.5,
                         edgecolor=deep_green,
                         facecolor='none')
ax2.add_patch(circle2)

ax2.scatter(*center2,
            c=deep_green,
            marker='x',
            s=160,
            linewidth=3,
            zorder=10,
            label='校正中心 $c^*$')

# 边界扩张标注
ax2.annotate('高权重引导边界扩张\n（降低误报）',
             xy=(tail_data[0][0], tail_data[0][1] + 0.1),
             xytext=(0.8, 2.2),
             arrowprops=dict(facecolor=deep_blue,
                             width=1.2,
                             headwidth=8),
             fontsize=12,
             color=deep_blue,
             ha='center',
             fontweight='bold')

# 中心偏移箭头
ax2.annotate('', xy=center2, xytext=(0, 0),
             arrowprops=dict(arrowstyle="->",
                             color='black',
                             linestyle='dashed',
                             lw=1.8))

ax2.text(0.35, -0.35,
         '中心校正偏移',
         fontsize=11,
         color='black')

ax2.legend(loc='lower left', frameon=True)

# =========================
# 高清矢量输出
# =========================
plt.tight_layout()
plt.savefig("传统损失函数与加权损失函数差异.pdf",
            format="pdf",
            bbox_inches='tight')

plt.show()