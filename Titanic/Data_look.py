import matplotlib as plt
import pandas as pd
df = pd.read_csv('train.csv')

#欠損値を中央値に
df["Age"].fillna(df.Age.median(), inplace=True)
#female = 0 , male = 1
SEX = df['Sex'].map({"female":0,"male":1}).astype(int)


split_data = []
for survived in[0,1]:
    split_data.append(df[df.Survived == survived])
temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked",bins= 3)
#plt.show()

temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16)
#plt.show()