import pandas as pd


data = pd.read_csv('output/pred_ans.csv', sep=',')



data.info()
#
# data2 = pd.read_csv('data/newsqa_combined-newsqa-data-v1.csv', sep=',')
# print(data2.story_text.str.len().sort_values())