import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
save_dir = r'D:\文档\大论文\MyLatex\fig\chapter4'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '消融实验M值折线图.pdf')

# 设置中文字体 (请确保系统有该字体，否则修改为其他支持中文的字体)
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据准备 =================
# X轴: M的值
M_values = [1, 3, 5, 10, 20]

# Y轴: F1-Score (对应 LaTeX 表 4-4 的数据)
f1_unsw = [0.8124, 0.8653, 0.8912, 0.9049, 0.9055]
f1_cic  = [0.8345, 0.9120, 0.9415, 0.9572, 0.9568]

# ================= 绘图逻辑 =================
plt.figure(figsize=(8, 5), dpi=100)

# 绘制折线
plt.plot(M_values, f1_unsw, marker='o', linestyle='--', linewidth=2, markersize=8, label='UNSW-NB15')
plt.plot(M_values, f1_cic,  marker='s', linestyle='-',  linewidth=2, markersize=8, label='CIC-IDS2017')

# 美化图表
plt.xlabel('聚类中心数量 ($M$)', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
# plt.title('聚类中心数量 M 对检测性能的影响', fontsize=14) # 论文中通常不需要图内标题

plt.xticks(M_values) # 强制显示所有M值的刻度
plt.ylim(0.75, 1.0)  # 设置Y轴范围
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)

# 在关键点上标注数值 (可选)
for x, y in zip(M_values, f1_cic):
    plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# ================= 保存 =================
print(f"正在保存图像至: {save_path}")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300)
print("保存成功！")

plt.show()