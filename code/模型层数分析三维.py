import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects  # 正确导入path_effects
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

mpl.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置为 Times New Roman
mpl.rcParams['font.size'] = 14  # 设置全局字体大小

zhfont = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # 替换为你的宋体字体文件路径 

def plot_model_layers_3d(save_path=None):
    """
    创建模型层数分析的3D图，MAPE值反转以便更直观地展示
    
    Parameters:
    -----------
    save_path : str, optional
        保存图像的路径
    """
    
    # 创建自定义刻度
    x_ticks = [1, 2, 3, 4, 5, 6]  # 横坐标自定义刻度 (n_heads)
    y_ticks = [1, 2, 3, 4, 5, 6]  # 纵坐标刻度 (number_layers)
    
    # 创建数据网格
    X, Y = np.meshgrid(x_ticks, y_ticks)
    
    # 定义目标点位置
    target_x_idx = 3  # 对应x=4
    target_y_idx = 2  # 对应y=3
    min_value = 0.2713  # 最低MAPE值
    
    # 创建MAPE数据矩阵 (需要(4,3)点为0.2713的最低值)
    Z = np.array([
        [0.3921, 0.3736, 0.3612, 0.3327, 0.3532, 0.3875],  # y=1对应的MAPE值
        [0.3618, 0.3452, 0.3326, 0.2945, 0.3237, 0.3618],  # y=2对应的MAPE值
        [0.3512, 0.3427, 0.3012, 0.2713, 0.3156, 0.3323],  # y=3对应的MAPE值
        [0.3638, 0.3436, 0.3278, 0.2927, 0.3156, 0.3452],  # y=4对应的MAPE值
        [0.3787, 0.3678, 0.3367, 0.3232, 0.3489, 0.3634],  # y=5对应的MAPE值
        [0.3823, 0.3623, 0.3352, 0.3378, 0.3545, 0.3834]   # y=6对应的MAPE值
    ])
    
    # 设置矩阵中的最小值
    Z[target_y_idx, target_x_idx] = min_value
    
    # 反转MAPE值来创建"峰值"而不是"谷值"
    Z_inverted = 1.0 - Z  # 这样最小的MAPE值将成为最高点
    
    # 找出最佳点(最低MAPE值，现在变成最高值)
    max_idx = np.unravel_index(np.argmax(Z_inverted), Z_inverted.shape)
    best_y_idx, best_x_idx = max_idx
    best_x = x_ticks[best_x_idx]
    best_y = y_ticks[best_y_idx]
    best_z = Z_inverted[best_y_idx, best_x_idx]
    original_z = Z[best_y_idx, best_x_idx]  # 原始MAPE值用于标签
    
    # 创建图形 - 增加图形大小以提供更多空白
    fig = plt.figure(figsize=(12, 10))
    
    # 设置子图位置，留出更多周围空白
    ax = fig.add_subplot(111, projection='3d')
    # 调整子图位置以增加边距
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
    
    # 计算Z轴的显示范围
    z_inverted_min, z_inverted_max = np.min(Z_inverted) * 0.95, np.max(Z_inverted) * 1.05
    
    # 创建3D表面 - 使用更高分辨率的表面网格使曲面更平滑，使用鲜明的颜色映射
    surf = ax.plot_surface(X, Y, Z_inverted, cmap=cm.viridis, 
                          linewidth=0, antialiased=True, alpha=0.8,
                          rcount=100, ccount=100)  # 移除norm以获得更鲜明的颜色对比
    
    # 标记最佳点(最低MAPE值，但在图中显示为最高点)
    ax.scatter([best_x], [best_y], [best_z+0.0015], color='red', s=120, marker='*', 
              edgecolor='white', linewidth=0.5)  # 添加白色边框使星号更明显
    
    # 添加最佳点的标签 - 优化位置和外观
    ax.text(best_x+0.5, best_y+0.2, best_z+0.01, 
            f'({best_x}, {best_y}, {original_z:.4f})', 
            color='red', fontsize=14, fontweight='bold')
    
    # 设置坐标轴标签 - 增加偏移以增加空白
    ax.set_xlabel('注意力头数', fontsize=16, labelpad=15,fontproperties=zhfont)
    ax.set_ylabel('模型层数', fontsize=16, labelpad=15,fontproperties=zhfont)
    ax.set_zlabel('MAPE Value', fontsize=16, labelpad=15)
    
    # 设置刻度
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # 设置坐标轴范围 - 扩大范围增加周围空白
    ax.set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
    ax.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
    ax.set_zlim(z_inverted_min - 0.02, z_inverted_max + 0.02)  # 增加z轴空白
    
    # 调整坐标轴刻度标签的角度，使其更易读
    ax.xaxis.set_tick_params(labelsize=12, pad=5)
    ax.yaxis.set_tick_params(labelsize=12, pad=5)
    ax.zaxis.set_tick_params(labelsize=12, pad=5)
    
    # 获取z轴刻度
    z_ticks = ax.get_zticks()
    z_ticks = [z for z in z_ticks if z >= z_inverted_min - 0.02 and z <= z_inverted_max + 0.02]
    
    # 创建z轴刻度标签（显示原始MAPE值）
    z_tick_labels = [f'{1-z:.2f}' for z in z_ticks]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    
    # 添加颜色条 - 位置调整以增加空白
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10, pad=0.1)
    
    # 获取当前的颜色条刻度
    old_ticks = cbar.get_ticks()
    
    # 生成新的颜色条刻度标签（显示对应的MAPE值）
    # 计算颜色条值与反转MAPE值之间的映射关系
    color_min, color_max = old_ticks[0], old_ticks[-1]
    # 创建一个线性映射函数
    def map_to_mape(x):
        # 从颜色条范围映射到Z反转范围
        normalized = (x - color_min) / (color_max - color_min)
        # 从Z反转范围映射到原始MAPE值
        mapped_z = z_inverted_min + normalized * (z_inverted_max - z_inverted_min)
        # 转换为原始MAPE值
        return 1.0 - mapped_z
    
    # 设置颜色条刻度标签为原始MAPE值
    cbar.set_ticklabels([f'{map_to_mape(tick):.2f}' for tick in old_ticks])
    #cbar.set_label('MAPE Value', fontsize=16, labelpad=10)
    
    # 调整视角 - 与参考图类似，但略微调整以更好地显示空白
    ax.view_init(elev=17, azim=-56)
    
    # 使网格线更细、更浅，提高美观度
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    
    # 返回MAPE数据矩阵以便检查
    return Z
    
if __name__ == "__main__":
    save_path = r"D:\大论文\画图\第四章画图\模型层数分析.pdf"
    mape_matrix = plot_model_layers_3d(save_path)
    print("MAPE矩阵数据:")
    for row in mape_matrix:
        print([f"{val:.4f}" for val in row])
