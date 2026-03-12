import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

mpl.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置为 Times New Roman
mpl.rcParams['font.size'] = 14  # 设置全局字体大小

# 替换为你的宋体字体文件路径 
zhfont = FontProperties(fname='C:/Windows/Fonts/simsun.ttc') 

def plot_param_sensitivity_3d(save_path=None):
    """
    创建参数敏感性分析的3D曲面图 (基于 F1-Score)，并极致消除保存时的留白
    """
    
    x_labels = [1, 3, 5, 7, 10, 15]      # 近邻数量 k
    y_labels = [0.5, 1.0, 2.0, 3.5, 5.0, 8.0] # 温度缩放因子 β
    
    X, Y = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))
    
    Z = np.array([
        [86.20, 89.10, 91.20, 90.50, 88.30, 85.10],  # β=0.5
        [88.10, 91.50, 92.80, 92.10, 89.60, 86.40],  # β=1.0
        [90.15, 93.60, 94.15, 93.85, 92.40, 91.05],  # β=2.0 (最高点)
        [89.50, 92.40, 93.20, 92.50, 91.10, 88.20],  # β=3.5
        [87.30, 90.10, 91.50, 90.60, 88.40, 85.50],  # β=5.0
        [82.10, 85.60, 87.60, 86.20, 84.10, 80.50]   # β=8.0
    ])
    
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    best_y_idx, best_x_idx = max_idx
    best_z = Z[best_y_idx, best_x_idx]
    
    # 调整画布大小，使其更紧凑 (可以根据需要调整比例)
    fig = plt.figure(figsize=(10, 8))
    
    # 使用紧凑的子图布局，消除默认边距
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # 关键修改1：边距全设为0
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                           linewidth=0, antialiased=True, alpha=0.85,
                           rcount=100, ccount=100)
    
    ax.scatter([best_x_idx], [best_y_idx], [best_z + 0.5], color='red', s=150, marker='*', 
               edgecolor='white', linewidth=0.8, zorder=5)
    
    ax.text(best_x_idx + 0.3, best_y_idx + 0.1, best_z + 1.0, 
            f'(k={x_labels[best_x_idx]}, β={y_labels[best_y_idx]}, {best_z:.2f}%)', 
            color='red', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('近邻数量 k', fontsize=16, labelpad=10, fontproperties=zhfont)
    ax.set_ylabel('温度缩放因子 β', fontsize=16, labelpad=10, fontproperties=zhfont)
    ax.set_zlabel('F1-Score (%)', fontsize=16, labelpad=10)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    
    ax.set_zlim(np.min(Z) - 1.0, np.max(Z) + 2.0)
    
    ax.xaxis.set_tick_params(labelsize=12, pad=0)
    ax.yaxis.set_tick_params(labelsize=12, pad=0)
    ax.zaxis.set_tick_params(labelsize=12, pad=0)
    
    # 调整颜色条的位置和大小，使其不占用过多外部空间
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.05)
    cbar.ax.tick_params(labelsize=12)
    
    # 放大 3D 图像以填满包围盒 (适用于较新的 Matplotlib 版本)
    try:
        ax.set_box_aspect(None, zoom=1.15) # 关键修改2：画面放大1.15倍，挤占空白
    except:
        pass # 如果版本较低不支持则跳过
        
    ax.view_init(elev=20, azim=-55)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if save_path:
        # 关键修改3：bbox_inches='tight' 和 pad_inches=0 组合，极致裁剪白边
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02) 
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    
    return Z

if __name__ == "__main__":
    save_path = r"D:\文档\大论文\MyLatex\fig\chapter3\param_sensitivity.pdf"
    f1_matrix = plot_param_sensitivity_3d(save_path)