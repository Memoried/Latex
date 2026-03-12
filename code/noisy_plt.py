import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
# 设置随机种子以保证每次生成相同的图像
np.random.seed(42)

# x 轴数据：生成 1000 个点
x = np.linspace(0, 100, 1000)

# 基础波动：使用正弦波模拟图中的周期性起伏 (大约 4-5 个波峰)
base_wave = 2.5 * np.sin(x * 0.3)

# 随机噪声：添加高斯噪声使线条呈现锯齿状
noise = np.random.normal(0, 1.5, 1000)

# 组合信号
y = base_wave + noise

# 添加显著的尖峰 (Anomaly/Spike)
# 在大约 38% 的位置添加一个向上的巨大尖峰
spike_index = int(1000 * 0.38)
y[spike_index] += 15

# 2. 绘制图像
fig, ax = plt.subplots(figsize=(10, 4)) # 设置画布比例，类似于原图的宽屏比例

# 绘制蓝色折线，稍微调整线宽
ax.plot(x, y, color='blue', linewidth=1.2)

# 3. 样式调整 (模仿原图的极简风格)
# 隐藏坐标轴刻度和标签
ax.set_xticks([])
ax.set_yticks([])

# 设置边框颜色为浅灰色，类似于原图的细边框
for spine in ax.spines.values():
    spine.set_color('#cccccc')
    spine.set_linewidth(1)

# 设置轴的显示范围，让线条填满整个宽度
ax.set_xlim(x.min(), x.max())
# 根据生成的最大最小值稍微留白
ax.set_ylim(y.min() - 2, y.max() + 2)

# 4. 保存为矢量图
# format='svg' 确保输出为矢量图，bbox_inches='tight' 移除多余的边缘空白
output_filename = 'noisy_signal_plot.svg'
plt.savefig(output_filename, format='svg', bbox_inches='tight')

print(f"矢量图已成功保存为: {output_filename}")

# 如果需要在运行代码时预览图像，可以取消注释下面这行
plt.show()