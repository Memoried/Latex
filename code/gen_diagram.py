import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

def draw_architecture_diagram():
    # 1. 设置画布和中文字体
    plt.figure(figsize=(20, 10), dpi=300)
    ax = plt.gca()
    
    # 设置支持中文的字体 (Windows下优先尝试微软雅黑或黑体)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 隐藏坐标轴
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # ==========================
    # 定义绘图辅助函数
    # ==========================
    def draw_box(x, y, w, h, color, text, subtext="", shape='rect', title_box=False):
        # 绘制矩形背景
        if shape == 'cylinder': # 简易模拟圆柱体（数据库）
            box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                         linewidth=1, edgecolor='#888888', facecolor=color)
            # 加两条横线模拟圆柱
            ax.add_patch(patches.Ellipse((x+w/2, y+h), w, h*0.2, facecolor='#DDDDDD', edgecolor='#888888'))
        elif shape == 'cloud': # 简易模拟云
            box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=2", 
                                         linewidth=1, edgecolor='#888888', facecolor=color)
        else: # 默认圆角矩形
            style = "round,pad=0.3" if not title_box else "square,pad=0"
            edge = '#888888' if not title_box else 'none'
            box = patches.FancyBboxPatch((x, y), w, h, boxstyle=style, 
                                         linewidth=1.5, edgecolor=edge, facecolor=color)
        
        ax.add_patch(box)
        
        # 添加文字
        cx, cy = x + w/2, y + h/2
        plt.text(cx, cy+0.5, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')
        if subtext:
            plt.text(cx, cy-1.2, subtext, ha='center', va='center', fontsize=9, color='#555555')
        return (x, y, w, h)

    def draw_layer_bg(x, y, w, h, label):
        # 绘制虚线背景框表示“层”
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#AAAAAA', 
                                 facecolor='#F9F9F9', linestyle='--', zorder=0)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h - 1.5, label, ha='center', va='center', 
                 fontsize=12, fontweight='bold', color='#555555', zorder=1)

    def draw_arrow(x1, y1, x2, y2, label="", curve=0.0):
        # 绘制箭头
        style = "Simple, tail_width=0.5, head_width=4, head_length=4"
        arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={curve}", 
                                color='#555555', arrowstyle='-|>', mutation_scale=15, lw=1.5)
        ax.add_patch(arrow)
        # 箭头上的文字
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            # 如果有曲线，文字位置稍微偏移
            if curve != 0: my += curve * 2
            plt.text(mx, my+0.8, label, ha='center', va='center', 
                     fontsize=9, backgroundcolor='white', color='#0000AA')

    # ==========================
    # 1. 绘制背景层 (Layers)
    # ==========================
    # 数据采集层
    draw_layer_bg(2, 5, 15, 40, "数据采集层")
    # 数据传输层
    draw_layer_bg(20, 5, 12, 40, "数据传输层")
    # 分析检测层 (核心)
    draw_layer_bg(35, 5, 38, 40, "分析检测层 (核心)")
    # 数据存储层
    draw_layer_bg(76, 5, 10, 40, "数据存储层")
    # 应用表现层
    draw_layer_bg(89, 5, 10, 40, "应用表现层")

    # ==========================
    # 2. 绘制具体组件 (Nodes)
    # ==========================
    
    # --- 采集层 ---
    draw_box(4, 30, 11, 4, '#E6F2FF', "企业网络", "镜像端口", shape='cloud')
    draw_box(4, 15, 11, 4, '#E6F2FF', "Zeek 探针", "协议解析 & 日志")

    # --- 传输层 ---
    draw_box(21, 22, 10, 6, '#E6FFE6', "Kafka 集群", "Topic: raw-traffic")

    # --- 分析层 (FastAPI 容器背景) ---
    container_rect = patches.Rectangle((37, 7), 34, 34, linewidth=1.5, edgecolor='#DDDDDD', 
                                       facecolor='white', linestyle='-', zorder=1)
    ax.add_patch(container_rect)
    plt.text(54, 39, "FastAPI 推理服务容器", ha='center', fontsize=10, color='#888888')

    # 分析层内部组件
    # 预处理
    draw_box(39, 22, 8, 5, '#FFE6E6', "预处理", "Z-Score / 特征筛选")
    
    # 两个模型 (并行)
    draw_box(51, 28, 10, 5, '#FFCCCC', "IED-Deep SVDD", "针对长尾分布")
    draw_box(51, 15, 10, 5, '#FFCCCC', "多模式聚类", "针对未知攻击")
    
    # 聚合
    draw_box(65, 22, 5, 5, '#FFE6E6', "聚合", "JSON格式化")

    # --- 存储层 ---
    draw_box(77, 20, 8, 10, '#FFFFE0', "Elasticsearch", "索引: logs/alerts", shape='cylinder')

    # --- 表现层 ---
    draw_box(90, 20, 8, 8, '#F0E6FF', "运维大屏", "Vue.js + ECharts")

    # ==========================
    # 3. 绘制连线 (Edges)
    # ==========================
    
    # 网络 -> Zeek
    draw_arrow(9.5, 30, 9.5, 19, "原始流量")
    
    # Zeek -> Kafka
    draw_arrow(15, 17, 21, 25, "JSON日志")
    
    # Kafka -> 预处理
    draw_arrow(31, 25, 39, 24.5, "消费数据")
    
    # 预处理 -> 两个模型
    draw_arrow(47, 24.5, 51, 30.5, "")
    draw_arrow(47, 24.5, 51, 17.5, "")
    
    # 两个模型 -> 聚合
    draw_arrow(61, 30.5, 65, 24.5, "")
    draw_arrow(61, 17.5, 65, 24.5, "")
    
    # 聚合 -> ES
    draw_arrow(70, 24.5, 77, 25, "持久化存储")
    
    # 大屏 <-> 聚合 (API) - 曲线
    draw_arrow(90, 26, 70, 27, "REST/WebSocket API", curve=0.3)
    
    # 大屏 -> ES (历史查询) - 曲线
    draw_arrow(90, 22, 85, 25, "历史查询", curve=-0.1)

    # 标题
    plt.text(50, 48, "基于流式处理的异常检测系统总体架构", ha='center', fontsize=16, fontweight='bold')

    # 保存
    plt.savefig('system_architecture_matplotlib.pdf', bbox_inches='tight', pad_inches=0.1)
    print("成功生成图片：system_architecture_matplotlib.pdf")
    plt.show()

if __name__ == "__main__":
    draw_architecture_diagram()