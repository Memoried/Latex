import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 1. 环境与字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

# --- 2. 绘图辅助函数 ---

def draw_tensor_stack(ax, xy, w, h, color='#fff9c4'):
    """绘制输入数据的张量堆叠示意"""
    x, y = xy
    # 模拟三层数据堆叠
    offset = 0.15
    for i in range(3):
        rect = patches.Rectangle((x + i*offset, y + i*offset), w, h, linewidth=1, edgecolor='black', facecolor=color, zorder=10-i)
        ax.add_patch(rect)
    return (x + w + 3*offset, y + h/2)

def draw_neural_net(ax, xy, w, h):
    """绘制简单的全连接神经网络示意"""
    x, y = xy
    layers = [3, 4, 3] # 节点数
    x_spacing = w / (len(layers) + 0.5)
    
    # 存储节点坐标
    node_coords = []
    for l_idx, n_nodes in enumerate(layers):
        layer_nodes = []
        y_spacing = h / (n_nodes + 1)
        curr_x = x + (l_idx + 0.5) * x_spacing
        for n_idx in range(n_nodes):
            curr_y = y + (n_idx + 1) * y_spacing
            circle = patches.Circle((curr_x, curr_y), 0.12, facecolor='white', edgecolor='#0277bd', zorder=20)
            ax.add_patch(circle)
            layer_nodes.append((curr_x, curr_y))
        node_coords.append(layer_nodes)
    
    # 绘制连线
    for i in range(len(node_coords)-1):
        for start_node in node_coords[i]:
            for end_node in node_coords[i+1]:
                line = patches.ConnectionPatch(xyA=start_node, xyB=end_node, coordsA="data", coordsB="data",
                                               color='#0277bd', alpha=0.3, linewidth=0.5, zorder=15)
                ax.add_patch(line)

def draw_knn_schematic(ax, xy, w, h):
    """绘制中心点与其邻居的连线 (IED示意)"""
    x, y = xy
    center = (x + w/2, y + h/2)
    neighbors = [
        (center[0] - 0.3, center[1] + 0.3),
        (center[0] + 0.4, center[1] + 0.2),
        (center[0] - 0.1, center[1] - 0.35)
    ]
    
    # 绘制中心点
    ax.add_patch(patches.Circle(center, 0.1, color='red', zorder=20))
    # 绘制邻居和连线
    for nb in neighbors:
        ax.add_patch(patches.Circle(nb, 0.08, color='gray', zorder=20))
        ax.plot([center[0], nb[0]], [center[1], nb[1]], color='red', linestyle='--', linewidth=1, zorder=15)
    
    ax.text(x+0.1, y+0.1, "k-NN", fontsize=8, color='gray')

def draw_exp_curve(ax, xy, w, h):
    """绘制指数权重映射曲线"""
    x, y = xy
    # 坐标轴
    ax.arrow(x+0.2, y+0.2, w-0.4, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(x+0.2, y+0.2, 0, h-0.4, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # 曲线
    X = np.linspace(0, w-0.6, 20)
    Y = 0.1 * np.exp(2 * X) # 模拟指数
    # 归一化Y以适应方框
    Y = Y / Y.max() * (h-0.7)
    ax.plot(x + 0.2 + X, y + 0.2 + Y, color='#e65100', linewidth=2)
    ax.text(x+w-0.5, y+0.3, "High", fontsize=7, color='#e65100')

def draw_cluster_center(ax, xy, w, h):
    """绘制特征中心 C"""
    x, y = xy
    cx, cy = x + w/2, y + h/2
    # 靶心图标
    ax.add_patch(patches.Circle((cx, cy), 0.35, color='#1a237e', alpha=0.2))
    ax.add_patch(patches.Circle((cx, cy), 0.2, color='white'))
    ax.add_patch(patches.Circle((cx, cy), 0.1, color='#1a237e'))
    # 十字准星
    ax.plot([cx-0.4, cx+0.4], [cy, cy], color='#1a237e', linewidth=1)
    ax.plot([cx, cx], [cy-0.4, cy+0.4], color='#1a237e', linewidth=1)

# --- 3. 主绘图逻辑 ---
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# === 定义布局坐标 ===
box_props = dict(boxstyle="round,pad=0.02", ec="black", lw=1.5)

# 1. 输入 X
pos_input = (0.5, 4.0)
draw_tensor_stack(ax, pos_input, 0.8, 1.2)
ax.text(1.0, 3.8, "原始流量\nBatch X", ha='center', fontsize=11, fontweight='bold')

# 箭头 Input -> Encoder
ax.annotate("", xy=(2.5, 4.6), xytext=(1.6, 4.6), arrowprops=dict(arrowstyle="->", lw=2))

# 2. 编码器 Encoder
rect_enc = patches.FancyBboxPatch((2.5, 3.6), 2.5, 2.0, boxstyle="round,pad=0.1", fc='#e3f2fd', ec='#1565c0', lw=2)
ax.add_patch(rect_enc)
draw_neural_net(ax, (2.5, 3.6), 2.5, 2.0)
ax.text(3.75, 5.8, "特征编码器 $\phi(\\cdot)$", ha='center', fontsize=12, fontweight='bold', color='#0d47a1')

# 箭头 Encoder -> Z
ax.annotate("", xy=(6.0, 4.6), xytext=(5.1, 4.6), arrowprops=dict(arrowstyle="->", lw=2))

# 3. 特征空间 Z
rect_z = patches.FancyBboxPatch((6.0, 3.6), 1.5, 2.0, boxstyle="round,pad=0.1", fc='#e0f2f1', ec='#00695c', lw=2)
ax.add_patch(rect_z)
# 画一些随机点
rnd_x = np.random.uniform(6.2, 7.3, 10)
rnd_y = np.random.uniform(3.8, 5.4, 10)
ax.scatter(rnd_x, rnd_y, s=30, c='#00695c', alpha=0.6)
ax.text(6.75, 3.3, "隐特征 $Z$", ha='center', fontsize=11, fontweight='bold', color='#004d40')

# === 分支路径 ===

# 路径A：向上 -> 距离计算
ax.annotate("", xy=(8.5, 6.5), xytext=(7.6, 4.8), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=2))

# 4. 特征中心 C (独立模块)
rect_c = patches.FancyBboxPatch((8.0, 7.5), 1.5, 1.2, boxstyle="circle,pad=0.1", fc='#e8eaf6', ec='#283593', lw=2)
# draw_cluster_center(ax, (8.5, 7.5), 1.0, 1.0) # 这里简化直接用圆形
ax.add_patch(patches.Circle((9.25, 8.1), 0.6, fc='#e8eaf6', ec='#283593', lw=2))
draw_cluster_center(ax, (8.75, 7.6), 1.0, 1.0)
ax.text(9.25, 8.9, "特征中心 $c$", ha='center', fontsize=11, fontweight='bold', color='#1a237e')

# 箭头 C -> 距离计算
ax.annotate("", xy=(10.0, 7.2), xytext=(9.7, 7.8), arrowprops=dict(arrowstyle="->", lw=1.5))

# 5. 距离计算模块
rect_dist = patches.FancyBboxPatch((10.0, 6.0), 2.5, 1.5, boxstyle="round,pad=0.1", fc='#f3e5f5', ec='#7b1fa2', lw=2)
ax.add_patch(rect_dist)
ax.text(11.25, 6.75, "距离计算\n$\|z_i - c\|^2$", ha='center', fontsize=11, fontweight='bold', color='#4a148c')

# 路径B：向下 -> 多样性评估
ax.annotate("", xy=(8.5, 2.0), xytext=(7.6, 4.4), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2))

# 6. IED 评估模块
rect_ied = patches.FancyBboxPatch((8.5, 1.0), 2.0, 2.0, boxstyle="round,pad=0.1", fc='#fff3e0', ec='#ef6c00', lw=2)
ax.add_patch(rect_ied)
draw_knn_schematic(ax, (8.5, 1.2), 2.0, 1.8)
ax.text(9.5, 3.2, "IED 多样性评估", ha='center', fontsize=11, fontweight='bold', color='#e65100')

# 箭头 IED -> 权重
ax.annotate("", xy=(11.5, 2.0), xytext=(10.6, 2.0), arrowprops=dict(arrowstyle="->", lw=2))

# 7. 权重生成模块
rect_w = patches.FancyBboxPatch((11.5, 1.0), 2.0, 2.0, boxstyle="round,pad=0.1", fc='#ffe0b2', ec='#e65100', lw=2)
ax.add_patch(rect_w)
draw_exp_curve(ax, (11.5, 1.2), 2.0, 1.8)
ax.text(12.5, 3.2, "权重映射 $w_i$", ha='center', fontsize=11, fontweight='bold', color='#bf360c')


# === 汇聚 ===

# 8. 总损失函数
rect_loss = patches.FancyBboxPatch((13.5, 3.5), 2.0, 2.0, boxstyle="round,pad=0.1", fc='#ffebee', ec='#c62828', lw=2)
ax.add_patch(rect_loss)
ax.text(14.5, 4.8, "加权损失\n$\mathcal{L}_{total}$", ha='center', fontsize=12, fontweight='bold', color='#b71c1c')
ax.text(14.5, 4.0, "$\sum w_i \cdot d_i$", ha='center', fontsize=10, color='#b71c1c')

# 连接线
ax.annotate("", xy=(14.5, 5.6), xytext=(12.6, 6.5), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10", lw=2))
ax.annotate("", xy=(14.5, 3.4), xytext=(13.6, 2.0), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=-90,rad=10", lw=2))


# === 反馈闭环 ===

# 梯度反向传播 (红虚线)
ax.annotate("", xy=(3.75, 3.5), xytext=(14.5, 3.4), 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5", color='#d32f2f', linestyle='--', lw=2))
ax.text(9.0, 0.2, "梯度反向传播 (更新 $\Theta$)", ha='center', fontsize=11, fontweight='bold', color='#d32f2f', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# 中心校正 (蓝虚线)
ax.annotate("", xy=(9.85, 8.1), xytext=(14.5, 5.6), 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4", color='#1565c0', linestyle='--', lw=2))
ax.text(12.0, 8.5, "中心校正 (Weighted Avg)", ha='center', fontsize=10, fontweight='bold', color='#1565c0', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.title("图 3-2 基于实例多样性加权(IED)的系统架构详图", fontsize=16, y=0.98, fontweight='bold')
plt.tight_layout()
plt.show()