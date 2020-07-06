import pandas as pd
import glob

path = '../prediction_v2'
all_files = glob.glob(path + "/**/*.csv", recursive=True)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'index' in df.columns:
        df = df.drop(['index'], axis=1)

    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)
# data = data.sort_values(by=['idx']).reset_index()

# data = data.drop(['idx', 'index'], axis=1)
if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)
if 'Unnamed: 0.1' in data.columns:
    data = data.drop(['Unnamed: 0.1'], axis=1)

data.info()
# data.to_csv('../output/newsqa_w_para_v2.csv')
data.to_csv('../output/pred_ans_v2.csv')