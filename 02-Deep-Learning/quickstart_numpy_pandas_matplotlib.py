# -*- coding: utf-8 -*-
"""
NumPy + Matplotlib + Pandas 半天速览（带详细解释）
===================================================
每个操作都附有中文解释，帮助你理解"这行代码在干什么"以及"深度学习里哪里会用到它"。

建议学习方式：
  1. 按顺序读，每读完一个知识点就运行一次
  2. 重点理解 NumPy 的矩阵运算（深度学习就是矩阵运算）
  3. 修改变量值，观察输出变化，加深理解
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# ============================================================
#
#   第一部分：NumPy  ——  深度学习一切数据的基础
#
#   为什么重要？
#   神经网络中的所有数据都是 张量（Tensor），而 NumPy 的多维数组
#   就是 PyTorch 张量的"前身"。PyTorch 的 API 几乎和 NumPy 一样。
#   学会了 NumPy，就学会了 80% 的 PyTorch 张量操作。
#
#   深度学习对应关系：
#     np.array()      →  torch.tensor()      数据容器
#     np.random.randn →  torch.randn          初始化权重
#     A @ B           →  torch.matmul(A, B)   全连接层 / Attention 的核心运算
#     x.sum(axis=0)   →  x.sum(dim=0)         求 loss / 归一化 / Softmax 都需要
#     x.reshape()     →  x.view() / x.reshape 改形状，处处要用
#
# ============================================================

# --- 1. 创建数组（相当于 PyTorch 中创建 Tensor）---
print("=" * 50)
print("NumPy 基础")
print("=" * 50)

# np.array()     直接用 Python 列表创建数组
a = np.array([1, 2, 3, 4, 5])

# np.zeros()     创建全 0 数组
# 参数 (3,4) 是形状（shape）：3 行 × 4 列
# 用途：初始化偏置项（bias）、占位数组
b = np.zeros((3, 4))

# np.ones()      创建全 1 数组
# 用途：初始化某些权重、创建掩码
c = np.ones((2, 3))

# np.arange(起点, 终点, 步长)   类似 Python 的 range，但返回数组
# 用途：生成 epoch 编号、索引序列
d = np.arange(0, 10, 2)          # → [0, 2, 4, 6, 8]

# np.random.randn(d0, d1, ...)   从标准正态分布 N(0,1) 采样
# ★ 深度学习中最重要的初始化方式之一！
#   Xavier 初始化、Kaiming 初始化都是基于 randn 的变体
e = np.random.randn(3, 3)        # 3×3 矩阵，每个元素 ~ N(0,1)

# np.linspace(起点, 终点, 个数)   在区间内均匀取 N 个点
# 用途：生成一组实验参数（如学习率 0.001 → 0.1 的对数刻度）
f = np.linspace(0, 1, 5)         # → [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"a = {a}")
print(f"e (服从标准正态分布的随机矩阵) =\n{e}")
# shape 属性返回数组的形状（元组）；dtype 返回元素的数据类型
print(f"e.shape = {e.shape}, e.dtype = {e.dtype}")
# 输出示例：e.shape = (3, 3), e.dtype = float64
# float64 是 NumPy 默认的浮点类型；PyTorch 默认用 float32（省显存）


# --- 2. 索引与切片 —— 从数组中取你需要的部分 ---
print("\n--- 索引与切片 ---")

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"x =\n{x}")

# 索引语法：[行, 列]
# 注意：索引从 0 开始
print(f"x[0, 1] = {x[0, 1]}")       # 第 0 行、第 1 列 → 2

# 切片语法：start:stop（不包含 stop）
# 冒号单独出现表示"全部"
print(f"x[:, 0] = {x[:, 0]}")       # : 表示所有行，0 表示第 0 列 → [1, 4, 7]

# 混合使用：行取 [1:]（从第1行到末尾），列取 [:2]（前两列）
print(f"x[1:, :2] =\n{x[1:, :2]}")  # → [[4,5], [7,8]]

# ★ 布尔索引：用条件表达式筛选数据
# 深度学习场景：找出 loss 大于阈值的样本、筛选预测正确的索引
mask = x > 4                        # mask 是一个 True/False 矩阵
print(f"mask (x > 4 的布尔矩阵):\n{mask}")
print(f"满足条件的元素: {x[mask]}")  # 只返回 True 位置的元素 → [5,6,7,8,9]


# --- 3. 矩阵运算（深度学习的核心运算）★ 最重要 ---
print("\n--- 矩阵运算（深度学习核心）---")

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# ☆ 逐元素运算（element-wise）：对应位置分别计算
print(f"A + B =\n{A + B}")           # 对应位置相加
print(f"A * B =\n{A * B}")           # 对应位置相乘（★ 不是矩阵乘法！）

# ☆ 矩阵乘法（matrix multiplication）：这才是线性代数里的矩阵乘法
#   神经网络的每一层本质上就是：output = input @ weight + bias
#   一个全连接层的计算：hidden = X @ W  （X: [batch, in_dim], W: [in_dim, out_dim]）
print(f"A @ B  (矩阵乘法)=\n{A @ B}")            # @ 是 Python 3.5+ 的矩阵乘法运算符
print(f"np.dot(A, B) (也是矩阵乘法)=\n{np.dot(A, B)}")  # 旧写法，效果相同

# ☆ 广播（Broadcasting）—— NumPy/PyTorch 最精妙的设计
# 当两个数组 shape 不同时，NumPy 会自动扩展较小的数组
# 规则：从右往左对齐维度，不匹配的维度必须有一个是 1（或不存在）
# 深度学习场景：batch 数据 + 偏置；Attention 中的 mask 加法
broadcast_result = np.ones((3, 1)) + np.ones((1, 4))
print(f"\n广播示例: (3,1) + (1,4) → 结果的 shape = {broadcast_result.shape}")
# 结果是一个 (3,4) 的矩阵，全是 2
# 原理：np.ones((3,1)) → 在列方向复制 4 次变成 (3,4)
#       np.ones((1,4)) → 在行方向复制 3 次变成 (3,4)
#       然后逐元素相加


# --- 4. 常用统计函数 ---
print("\n--- 常用统计函数 ---")

v = np.array([1, 2, 3, 4])
print(f"原数组 v = {v}")
print(f"sum 求和 = {v.sum()}")
print(f"mean 均值 = {v.mean():.2f}")
print(f"std 标准差 = {v.std():.2f}")    # 衡量数据的离散程度
print(f"argmax 最大值位置 = {v.argmax()} (索引从0开始)")
print(f"max 最大值 = {v.max()}")

# ★ 沿指定轴（axis）操作 —— 理解 axis 对深度学习非常重要
#   axis=0 → 沿着行的方向（竖直向下）→ 跨行操作 → 结果是对每"列"计算
#   axis=1 → 沿着列的方向（水平向右）→ 跨列操作 → 结果是对每"行"计算
#   助记：axis=N 意味着"消除第 N 个维度"
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"\nm =\n{m}")
print(f"m.sum(axis=0) = {m.sum(axis=0)}  ← 跨行求和（每列汇总）→ (3,)")
print(f"m.sum(axis=1) = {m.sum(axis=1)}  ← 跨列求和（每行汇总）→ (2,)")
# 深度学习场景：Softmax(axis=-1) 对最后一维做归一化
#              Loss 计算时对 batch 维度求 mean


# --- 5. reshape（变形）与 transpose（转置）---
print("\n--- reshape 与 transpose ---")

# reshape: 不改变数据，只改变"看"数据的视角
# 前提：新形状的元素总数必须和原来相同
t = np.arange(12).reshape(3, 4)      # 0~11 一共 12 个数 → 3行×4列
print(f"arange(12).reshape(3,4):\n{t}")

# .T 或 .transpose()：矩阵转置，行列互换
print(f"t.T (转置):\n{t.T}")           # 3×4 → 4×3

# flatten()：把任意维度的数组拍平成一维
# 用途：全连接层之前，需要把 CNN 输出的多维特征图展平
print(f"t.flatten(): {t.flatten()}")  # → [0,1,2,...,11]

# reshape 中的 -1：让 NumPy 自动推断该维度的大小
# 用途：你只知道想变成多少列，让 NumPy 自己算行数
print(f"reshape(-1, 1):\n{t.reshape(-1, 1)}")  # 变成 12 行 1 列


# ============================================================
#
#   第二部分：Matplotlib  ——  可视化你的数据和训练曲线
#
#   为什么重要？
#   深度学习不是"黑盒"——你需要通过图表来：
#     1. 看 loss 曲线判断模型是否在收敛
#     2. 可视化数据分布，判断是否需要预处理
#     3. 展示模型预测结果
#
#   PyTorch 生态里常用的替代方案：TensorBoard、wandb（更强大）
#   但 matplotlib 是最基础、最通用的，论文里的图 90% 用它画
#
# ============================================================

print("\n" + "=" * 50)
print("Matplotlib 基础（图表已保存到文件中）")
print("=" * 50)

# --- 1. 折线图 —— 训练过程中最重要的一张图 ---
# 横轴：epoch（训练轮数）
# 纵轴：loss（损失值，衡量模型预测与真实值的差距）
epochs = np.arange(1, 21)                                          # 1~20 轮
# 模拟 loss 下降曲线：理想情况下 loss 随训练递减（这里用 1/√epoch 模拟趋势）
train_loss = 2.0 / np.sqrt(epochs) + np.random.randn(20) * 0.1    # 训练集 loss + 噪声
val_loss   = 2.0 / np.sqrt(epochs) + 0.1 + np.random.randn(20) * 0.08  # 验证集 loss（稍高）

# plt.figure(figsize=(宽, 高))  创建一张画布，单位为英寸
plt.figure(figsize=(10, 4))

# plt.subplot(行, 列, 位置)  在一张画布上画多个子图
# 这里 1 行 2 列，当前画第 1 个
plt.subplot(1, 2, 1)

# plt.plot(x, y, '颜色-标记-线型', label='图例名')
# 'b-o' = blue(蓝) + 实线 + circle(圆圈标记)
# 'r-s' = red(红)  + 实线 + square(方块标记)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
plt.plot(epochs, val_loss,   'r-s', label='Val Loss',   markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练曲线（Training Curves）')
plt.legend()                    # 显示图例
plt.grid(True, alpha=0.3)       # 添加半透明网格线

# --- 2. 散点图 —— 可视化数据分布 ---
# 场景：你拿到一份数据，想看看两个类别是否"线性可分"
plt.subplot(1, 2, 2)

np.random.seed(42)  # 固定随机种子，保证每次运行生成的"随机"数据相同

# 生成两个类别的二维数据（模拟特征空间）
# 每行是一个样本，两列是两个特征值
class_0 = np.random.randn(100, 2) + np.array([0, 0])     # 100个点，聚类中心在 (0,0)
class_1 = np.random.randn(100, 2) + np.array([3, 3])     # 100个点，聚类中心在 (3,3)

# scatter 参数说明：
#   前两个参数分别是 x 坐标数组和 y 坐标数组
#   c='颜色', alpha=透明度(0~1), s=点的大小, label=图例名
plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Class 0', s=20)
plt.scatter(class_1[:, 0], class_1[:, 1], c='red',  alpha=0.6, label='Class 1', s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('数据分布（散点图）')
plt.legend()

plt.tight_layout()  # 自动调整子图间距，避免标签重叠
plt.savefig('02-Deep-Learning/mpl_demo.png', dpi=100)  # 保存为 PNG 文件
print("图表已保存到 02-Deep-Learning/mpl_demo.png，请打开查看")
plt.close()         # 关闭画布，释放内存


# ============================================================
#
#   第三部分：Pandas  ——  处理表格数据
#
#   为什么重要？
#   深度学习项目中，数据很少是"干净的 numpy 数组"。
#   原始数据通常是 CSV / Excel 表格，需要 Pandas 来：
#     1. 读取 CSV / JSON / Excel
#     2. 数据清洗（去空值、去重复）
#     3. 统计分析、特征工程
#     4. 最后用 .values 转换成 NumPy 数组喂给模型
#
#   核心数据结构：
#     DataFrame：一张表格（有行索引、列名），类似 Excel 工作表
#     Series：   一列数据，类似 Excel 中的一列
#
# ============================================================

print("\n" + "=" * 50)
print("Pandas 基础")
print("=" * 50)

# --- 1. 创建 DataFrame（数据框）---
# 用字典创建：key = 列名，value = 该列的数据（列表）
df = pd.DataFrame({
    'name':    ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age':     [25, 30, 35, 28, 22],
    'score':   [88, 92, 85, 95, 78],
    'country': ['CN', 'US', 'CN', 'UK', 'US'],
})
print(f"原始数据（DataFrame）:\n{df}\n")
# 每一行是一个样本（row），每一列是一个特征（column/feature）


# --- 2. 基本操作 ---

# df.head(n)：查看前 n 行（默认 5 行）
# 场景：拿到一个几十万行的 CSV，先看一眼前几行确认数据格式
print(f"前3行 df.head(3):\n{df.head(3)}\n")

# df.describe()：数值列的统计摘要
# 场景：快速了解数据的均值、标准差、四分位数，检查是否有异常值
print(f"统计摘要 df.describe():\n{df.describe()}\n")

# 取单列：df['列名'] 返回一个 Series（类似一维数组）
print(f"df['score'] 这一列: {df['score'].values}")  # .values 得到 numpy 数组
# ★ 这就是 Pandas → NumPy 的桥梁：先用 Pandas 读数据，.values 转成数组喂给模型

# 条件筛选（类似 SQL 的 WHERE 子句）
print(f"筛选 score > 85 的行:\n{df[df['score'] > 85]}\n")
# 原理：
#   df['score'] > 85  →  返回一个布尔 Series  [True, True, False, True, False]
#   df[布尔Series]     →  只保留 True 对应的行


# --- 3. 分组聚合（类似 SQL 的 GROUP BY）---
# 场景：按国家统计平均成绩
print(f"按国家分组求平均分:\n{df.groupby('country')['score'].mean()}\n")
# 拆解：
#   df.groupby('country')          → 按 country 列分组
#   ['score']                      → 选 score 列
#   .mean()                        → 对每组求均值


# --- 4. 读写 CSV 文件（实际项目中的第一步和最后一步）---
csv_path = '02-Deep-Learning/sample_data.csv'

# 写入 CSV
df.to_csv(csv_path, index=False)  # index=False 不保存行号
print(f"已保存到 {csv_path}")

# 读取 CSV（★ 这是你以后最常用的 Pandas 命令 ★）
df_loaded = pd.read_csv(csv_path)
print(f"\n从 CSV 读回的数据:\n{df_loaded}")

# 补充常用操作（不需要运行，记下来就好）：
#   df.dropna()             删除含空值的行
#   df.fillna(0)            用 0 填充空值
#   df.drop_duplicates()    删除重复行
#   df['new_col'] = ...     新增一列
#   df.merge(other_df, on='key')   类似 SQL JOIN，合并两张表


# ============================================================
# 总结：三个库在深度学习项目中的分工
# ============================================================
print("\n" + "=" * 50)
print("总结：三个库在深度学习项目中的角色")
print("=" * 50)
print("""
  ┌──────────┐     ┌──────────┐     ┌───────────┐
  │  Pandas  │ ──► │  NumPy   │ ──► │  PyTorch  │
  │ 数据清洗  │     │ 数组运算  │     │ 模型训练   │
  └──────────┘     └──────────┘     └───────────┘
                                              │
  ┌──────────┐                               │
  │Matplotlib│ ◄─────────────────────────────┘
  │ 可视化    │     画出 loss 曲线、预测结果
  └──────────┘

  1. 用 Pandas 读取并清洗原始数据（CSV/Excel）
  2. df.values 转成 NumPy 数组
  3. NumPy 数组 → PyTorch Tensor → 喂给模型训练
  4. 用 Matplotlib 可视化训练过程和结果
""")

print("✅ 速览结束！你现在已经掌握了深度学习所需的数据处理基础。")
print("下一步：安装 PyTorch，开始吴恩达深度学习课程！")
