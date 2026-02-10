import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
save_dir = r'D:\文档\大论文\MyLatex\fig\chapter4'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '阈值敏感度分析.pdf')

# 设置中文字体
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据模拟 =================
# 模拟一个典型的 Precision-Recall 随阈值变化的过程
# X轴: 归一化阈值 [0.0, 1.0]
thresholds = np.linspace(0.05, 0.95, 19)

# 模拟数据生成逻辑：
# 阈值低 -> 判为异常多 -> Recall高(TP多), Precision低(FP多)
# 阈值高 -> 判为异常少 -> Recall低(FN多), Precision高(FP少)
recall    = 1.0 / (1.0 + np.exp(10 * (thresholds - 0.55))) # Sigmoid 变体模拟下降
precision = 1.0 / (1.0 + np.exp(-10 * (thresholds - 0.25))) # Sigmoid 变体模拟上升

# 修正一下边界，让数据看起来更真实一点
recall = np.clip(recall * 0.98 + 0.02, 0, 1)
precision = np.clip(precision * 0.98 + 0.02, 0, 1)

# 计算 F1 Score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# 找到 F1 的最大值及其对应的阈值
max_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[max_f1_idx]
best_f1 = f1_scores[max_f1_idx]

# ================= 绘图逻辑 =================
plt.figure(figsize=(8, 5), dpi=100)

# 绘制三条线
plt.plot(thresholds, precision, label='Precision (精确率)', color='#5B9BD5', linestyle='--', linewidth=2, marker='^', markersize=6)
plt.plot(thresholds, recall,    label='Recall (召回率)',    color='#ED7D31', linestyle='--', linewidth=2, marker='v', markersize=6)
plt.plot(thresholds, f1_scores, label='F1-Score',           color='#70AD47', linestyle='-',  linewidth=2.5, marker='o', markersize=7)

# 标注最佳点
plt.axvline(x=best_threshold, color='gray', linestyle=':', alpha=0.6)
plt.text(best_threshold + 0.02, best_f1, f'Optimal $\delta$={best_threshold:.2f}\nF1={best_f1:.3f}', 
         fontsize=10, verticalalignment='bottom', color='#333333')

# 美化图表
plt.xlabel('归一化判别阈值 ($\delta$)', fontsize=12)
plt.ylabel('性能指标', fontsize=12)
plt.xlim(0, 1.0)
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower center', fontsize=11, ncol=3, frameon=True)

plt.tight_layout()

# ================= 保存 =================
print(f"正在保存图像至: {save_path}")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300)
print("保存成功！")

plt.show()