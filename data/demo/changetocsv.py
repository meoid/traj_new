'''
此文件用于把邻接表数据转为csv，并且把第一列加上
'''

import pandas as pd

df = pd.read_csv('./edge_property.txt', sep=',', header=None, quotechar='"')
df.to_csv('./edge_property.csv', index=False, header=False)

print('done')
