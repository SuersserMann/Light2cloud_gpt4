import pandas as pd

# 读取csv文件，第一行设置为列名
df = pd.read_csv('input.csv', header=0)

# 将第一行替换为数字
df.columns = range(len(df.columns))

# 使用melt函数将数据透视成需要的格式
df = pd.melt(df, id_vars=0, var_name='new_col', value_name='value')

# 交换new_col和date列的位置，并将date列的值转换为日期格式
df['date'] = pd.to_datetime(df[0])
df = df[['new_col', 'date', 'value']]

# 重新设置列名
df.columns = ['new_col', 'date', 'value']

# 输出为csv文件
df.to_csv('new_filename.csv', index=False)
