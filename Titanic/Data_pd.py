import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv')
df = pd.DataFrame(df)
#print(df)

print(df.describe())
# 件数 (count)
# 平均値 (mean)
# 標準偏差 (std)
# 最小値(min)
# 第一四分位数 (25%)
# 中央値 (50%)
# 第三四分位数 (75%)
# 最大値 (max) を確認することができます。
print('dataframeの行数・列数の確認==>\n', df.shape)
print('indexの確認==>\n', df.index)
print('columnの確認==>\n', df.columns)
print('dataframeの各列のデータ型を確認==>\n', df.dtypes)



#今のデータにどれくらい欠損があるかをまずはざっと確認
#列単位で 欠損値NaN(not a number)が入っている個数をカウントする （正確には、isnull()でtrueが返ってくる個数をカウントしている）
print(df.isnull().sum())
# 1つでもNaNが含まれる行だけを抽出（最初の5行のみ表示）
print(df[df.isnull().any(axis=1)].shape)
print(df[df.isnull().any(axis=1)].head())


# 'payday'列にあるNaNを'0.0'に置き換える
#df.fillna(value={'payday': 0.0}, inplace=True)
#df.head()

# 'kcal'列にNaNがある行を削除する
#df.dropna(subset=['kcal'], axis=0, inplace=True)
#print(df.shape) # 207-166=41行のデータを削除した

# 'Embarked' 列の Nan '---' を に置き換える
df['Embarked'] = df['Embarked'].replace('', '---').astype(str)
#df.head()

# 'Name'はデータとして不要な気がするので、データから列ごと削除
df.drop(['Name'], axis=1, inplace=True)
#df.head()



#df['weather'].value_counts()

# groupbyメソッドで、'Sex'列ごとに'Survived'の数をカウントする
df.groupby(['Sex'])['Survived'].count()

# groupbyメソッドは複数列に対しても行える
# groupbyメソッドで、'month', 'period'列ごとに'sales'の数を合計する
#df.groupby(['month', 'period'])['sales'].sum()

# EmbarkedごとにSurvivedの平均値を出す
df.groupby(['Embarked'])['Survived'].mean()

# 前行との差分が欲しい時は .diff() を使う
#df['temperature_diff'] = df['temperature'].diff(periods=1)
#df[['temperature','temperature_diff']].head()

# ヒストグラム
df.plot(kind='hist', y='Age' , bins=10, figsize=(16,4), alpha=0.5)
#plt.show()
# 散布図
df.plot(kind='scatter', x='Age', y='Survived')
plt.show()