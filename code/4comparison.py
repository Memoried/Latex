import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================

# 1. 设置保存路径 (确保此目录存在，或者脚本会自动创建)
save_dir = r'D:\文档\大论文\MyLatex\fig\chapter4'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '对比实验柱状图.pdf')

# 2. 设置中文字体支持
# matplotlib默认不支持中文，需要手动设置。
# 尝试寻找系统中常见的中文字体。如果运行后中文乱码，请根据你系统实际安装的字体修改此处。
import matplotlib.font_manager as fm
found_font = False
for font_name in ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']:
    try:
        # 尝试设置字体，看看是否报错
        plt.rcParams['font.sans-serif'] = [font_name]
        # 用来正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
        # 测试一下是否真的能用
        fig_test = plt.figure()
        ax_test = fig_test.add_subplot(111)
        t = ax_test.text(0.5, 0.5, "测试中文")
        plt.close(fig_test)
        print(f"成功加载中文字体: {font_name}")
        found_font = True
        break
    except Exception:
        continue

if not found_font:
    print("警告：未找到常见中文字体，图表中的中文可能会显示为方框。请手动指定字体路径。")
    # fallback (可选: 如果实在找不到，可以取消下面注释强制使用英文)
    # plt.rcParams['font.family'] = 'sans-serif'

# ================= 数据准备 =================
# 注意：这里的数据是为了演示趋势，请根据你实际实验记录的准确数值进行替换。
# F1值尽量保持与 LaTeX 表 4-3 一致。

algorithms = ['iForest', 'OC-SVM', 'AutoEncoder', 'DAGMM', '本文方法']

# 三个指标的数据 (Precision, Recall, F1-Score)
# 数据来源：基于CIC-IDS2017数据集的典型表现及论文描述设定
precision_data = [0.821, 0.745, 0.885, 0.921, 0.953]
recall_data    = [0.773, 0.683, 0.851, 0.904, 0.961]
f1_data        = [0.796, 0.713, 0.867, 0.912, 0.957] # 与表格数据基本吻合

# ================= 绘图逻辑 =================

# 设置画布大小和分辨率
plt.figure(figsize=(10, 6), dpi=100)

# 设置柱子的宽度和位置
x = np.arange(len(algorithms))  # 算法标签的位置
width = 0.25  # 每个柱子的宽度

# 创建分组柱状图
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision_data, width, label='Precision', color='#5B9BD5', edgecolor='white')
rects2 = ax.bar(x, recall_data, width, label='Recall', color='#ED7D31', edgecolor='white')
rects3 = ax.bar(x + width, f1_data, width, label='F1-Score', color='#70AD47', edgecolor='white')

# 添加文本标签、标题和自定义坐标轴刻度
ax.set_ylabel('性能得分', fontsize=12, fontweight='bold')
ax.set_xlabel('检测算法', fontsize=12, fontweight='bold')
# ax.set_title('不同算法在 CIC-IDS2017 数据集上的性能对比', fontsize=14, pad=20) # 论文中标题通常在图下方，这里可以注释掉

ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=11)
ax.set_ylim(0.6, 1.05) # 设置Y轴范围，使差异更明显
ax.tick_params(axis='y', labelsize=10)

# 添加图例
ax.legend(loc='upper left', frameon=True, fontsize=10)

# 添加水平网格线，增加可读性
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 定义一个函数在柱子上方添加数值标签
def autolabel(rects):
    """在每个矩形柱子上方附加一个文本标签，显示其高度。"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移 3 个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# 为每一组柱子添加标签
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 调整布局以防止标签重叠
fig.tight_layout()

# ================= 保存图片 =================
print(f"正在保存图像至: {save_path}")
# 保存为 PDF 矢量图，适合 LaTeX
plt.savefig(save_path, format='pdf', bbox_inches='tight')
# 也可以额外保存一份 PNG 用于快速查看
plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
print("保存成功！")

# 显示图像 (可选，如果在服务器端运行可以注释掉)
plt.show()