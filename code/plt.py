import matplotlib.pyplot as plt
import numpy as np

# ======================
# CNNIC 数据（2023–2025）
# ======================
years = ["2023年", "2024年", "2025年"]

netizens = [10.92, 11.08, 11.23]      # 网民规模（亿）
penetration = [77.5, 78.6, 79.7]      # 互联网普及率（%）
base_stations = [337, 389, 455]       # 5G基站数量（万）

x = np.arange(len(years))

# ======================
# 学术论文风格设置（宋体·小四）
# ======================
plt.rcParams["font.sans-serif"] = ["SimSun"]   # 宋体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12        # 小四
plt.rcParams["axes.titlesize"] = 12   # 标题
plt.rcParams["axes.labelsize"] = 12   # 坐标轴标题
plt.rcParams["xtick.labelsize"] = 12  # x轴刻度
plt.rcParams["ytick.labelsize"] = 12  # y轴刻度
plt.rcParams["legend.fontsize"] = 12  # 图例


fig, ax1 = plt.subplots(figsize=(8,5))

# ======================
# 左轴：网民规模（柱状）
# ======================
ax1.bar(
    x - 0.15, netizens, width=0.3,
    color="#6b7c93",
    label="网民规模（亿人）"
)

# 折线：互联网普及率
ax1.plot(
    x, penetration, marker="o",
    linewidth=2, color="#2f3e4e",
    label="互联网普及率（%）"
)

ax1.set_ylabel("网民规模 / 普及率")
ax1.set_xlabel("年份")
ax1.set_xticks(x)
ax1.set_xticklabels(years)

# ======================
# 右轴：5G基站
# ======================
ax2 = ax1.twinx()
ax2.plot(
    x + 0.15, base_stations,
    linestyle="--", marker="s",
    color="#8d99ae",
    label="5G基站数量（万座）"
)
ax2.set_ylabel("5G基站数量（万）")

# ======================
# 合并图例
# ======================
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# ======================
# 样式
# ======================
ax1.grid(True, linestyle="--", alpha=0.3)
plt.title("中国互联网发展核心指标（2023–2025，CNNIC）")

plt.tight_layout()

# ======================
# 导出为矢量图
# ======================
# plt.savefig("cnnic_2025_trend.svg", format="svg")
plt.savefig("cnnic_2025_trend.pdf", format="pdf")

plt.show()
