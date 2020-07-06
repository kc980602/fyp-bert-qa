import pandas as pd

v2 = pd.read_csv('../output/pred_ans_v2.csv', sep=',')
v1 = pd.read_csv('../output/pred_ans.csv', sep=',')

# data = data.loc[data['para_text'].isnull()]

v2_x = v2.loc[(v2['is_question_bad'] != '1.0') | (v2['is_answer_absent'] <= 0.5)]
v1_x = v1.loc[(v1['is_question_bad'] != '1.0') | (v1['is_answer_absent'] <= 0.5)]

print(v2_x.shape[0], v1_x.shape[0])

v2 = v2[['story_id', 'question', 'is_question_bad','is_answer_absent']]
v1 = v1[['story_id', 'question', 'is_question_bad','is_answer_absent']]

v2.info()
v1.info()

print(v2.groupby(['story_id', 'question']).ngroups)
print(v1.groupby(['story_id', 'question']).ngroups)

data = pd.concat([v1,v2]).drop_duplicates(keep=False)


data.info()

# for row in data.iterrows():
#     print(row)

#
# duplicateDFRow = data_or[data_or.duplicated()]
# print(duplicateDFRow)

# data.to_csv('../output/newsqa_w_para_v2_null.csv')
