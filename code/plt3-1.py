import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# --- 1. 字体兼容性设置 ---
# 自动寻找系统中可用的中文黑体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 设置随机种子以保证实验图的可复现性
np.random.seed(42)

# --- 2. 模拟数据生成 ---
# 生成“头部”高频正常样本 (密集, 数量多)
n_head = 160
head_data = np.random.normal(loc=0.0, scale=0.35, size=(n_head, 2))

# 生成“尾部”长尾正常样本 (稀疏, 数量少, 分布在边缘)
tail_data = np.array([
    [1.6, 1.2], [1.4, -1.3], [-1.7, 1.0], 
    [-1.2, -1.6], [1.9, 0.1], [-0.4, 1.8]
])

# --- 3. 绘图初始化 ---
fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=120)
plt.subplots_adjust(wspace=0.3)

# === 左图：传统方法 (Standard Approach) ===
ax1 = axes[0]
ax1.set_title("(a) 传统方法 (受高频样本主导)", fontsize=14, pad=15, fontweight='bold')
ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_aspect('equal')

# 绘制数据点
ax1.scatter(head_data[:, 0], head_data[:, 1], c='gray', s=25, alpha=0.4, label='头部样本 (高频)')
ax1.scatter(tail_data[:, 0], tail_data[:, 1], c='gray', s=25, alpha=0.4, label='尾部样本 (稀疏)')

# 绘制偏置的中心和收缩的边界
center1 = (0, 0)
radius1 = 0.85
circle1 = patches.Circle(center1, radius1, linewidth=2, edgecolor='#d62728', facecolor='none', linestyle='--')
ax1.add_patch(circle1)
ax1.scatter(*center1, c='#d62728', marker='x', s=120, linewidth=3, zorder=10, label='偏置中心 $c$')

# 标注：精确率受损点 (误报)
ax1.annotate('误判为异常\n(精确率下降)', xy=tail_data[0], xytext=(1.8, 2.1),
             arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1, headwidth=7),
             fontsize=10, color='#d62728', ha='center', fontweight='bold')
ax1.legend(loc='lower left', fontsize=9)

# === 右图：多样性加权方法 (Proposed Method) ===
ax2 = axes[1]
ax2.set_title("(b) 提出的多样性加权方法", fontsize=14, pad=15, fontweight='bold')
ax2.set_xlim(-2.5, 2.5); ax2.set_ylim(-2.5, 2.5)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_aspect('equal')

# 绘制头部点 (权重低, 点小)
ax2.scatter(head_data[:, 0], head_data[:, 1], c='gray', s=25, alpha=0.3)
# 绘制尾部点 (权重高, 点大且亮)
weights = [180] * len(tail_data) 
ax2.scatter(tail_data[:, 0], tail_data[:, 1], c='#1f77b4', s=weights, alpha=0.8, 
            edgecolors='black', linewidth=1, label='加权样本 (高IED权重)')

# 绘制校正后的中心和扩张后的边界
center2 = (0.2, 0.15) # 模拟中心向稀疏区偏移
radius2 = 2.15
circle2 = patches.Circle(center2, radius2, linewidth=2.5, edgecolor='#2ca02c', facecolor='none', linestyle='-')
ax2.add_patch(circle2)
ax2.scatter(*center2, c='#2ca02c', marker='x', s=120, linewidth=3, zorder=10, label='校正中心 $c^*$')

# 标注：中心偏移和边界扩张
ax2.annotate('高权重引导边界扩张\n(降低误报)', xy=(tail_data[0][0], tail_data[0][1]+0.1), xytext=(0.8, 2.2),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1, headwidth=7),
             fontsize=10, color='#1f77b4', ha='center', fontweight='bold')

# 绘制偏移箭头
ax2.annotate('', xy=center2, xytext=(0,0), arrowprops=dict(arrowstyle="->", color='gray', linestyle='dashed', lw=1.5))
ax2.text(0.3, -0.25, '中心校正偏移', fontsize=9, color='gray', fontstyle='italic')

ax2.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig("传统损失函数与加权损失函数差异.pdf", format="pdf")
plt.show()