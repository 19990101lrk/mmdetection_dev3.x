import os
import csv
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# 创建两个空列表来存储所有文件的recall和precision数据
all_recalls = []
all_precisions = []
file_names=[]

path = 'E:/lrk/trail/logs/SAR/SSDD/PR_curve/FPN/'

files = os.listdir(path)
dfs = []
for file in files:
    if file.endswith('.xlsx'):
        file_path = os.path.join(path, file)
        df = pd.read_excel(file_path, header=None, names=['recall', 'precision'])
        dfs.append(df)

fig, ax = plt.subplots()
for file in files:
    if file.endswith('.xlsx'):
        file_path = os.path.join(path, file)
        df = pd.read_excel(file_path, header=None, names=['recall', 'precision'])
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(file)[0]
        ax.plot(df['recall'], df['precision'], label=file_name)

plt.rcParams['font.sans-serif'] = ['Times new Roman']  # 设置全部字体为Euclid
config = {
    "font.family": 'Times new Roman',  # 设置字体类型
    "font.size": 10,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('PR Curve')
ax.legend()  # 添加图例
plt.xlim(0.0,0.8)
plt.ylim(0.01,0.99)
plt.savefig(path + 'img/SSDD_Compare.svg', format='svg', dpi=600, bbox_inches='tight')
plt.show()



# # 绘制PR曲线
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
# for i in range(1):
#     # plt.plot(all_recalls[i], all_precisions[i], color=colors[i % len(colors)], label=f'File {i + 1}')
#     plt.plot(all_recalls[i], all_precisions[i], color=colors[i % len(colors)], label=file_names[i])
# plt.rcParams['font.sans-serif'] = ['Times new Roman']  # 设置全部字体为Euclid
# config = {
#     "font.family": 'Times new Roman',  # 设置字体类型
#     "font.size": 9,
# #     "mathtext.fontset":'stix',
# }
# rcParams.update(config)
# plt.ylim(0.825,1)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('P_R_Curve')
# plt.legend()
# plt.savefig(root + 'img/SSDD_Compare.svg', format='svg', dpi=600, bbox_inches='tight')
# plt.show()
#
