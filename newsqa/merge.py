import pandas as pd
import glob

path = '../prediction'
all_files = glob.glob(path + "/**/*.csv", recursive=True)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)

data = data.drop_duplicates(subset=['story_id', 'question'])
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
data.info()
data.to_csv('../output/pred_ans.csv')