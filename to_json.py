import pandas as pd
import json
from ast import literal_eval
import numpy as np

d_pred = pd.read_csv('./output/pred_ans_w_class.csv', sep=',')
d_pred = d_pred.fillna('')
d_pred_cross = pd.crosstab(d_pred['story_id'], d_pred['class'])

d_pred_cross = d_pred_cross.sort_values(by=[1], ascending=False)

d_pred_cross = d_pred_cross.head(10)

result = []

for story_id, idx in d_pred_cross.T.iteritems():
    d_question_full = d_pred.loc[d_pred['story_id'] == story_id]
    if d_question_full.shape[0]:
        d_question = d_question_full.drop(columns=['story_id', 'story_text'])
        one_row = d_question_full.iloc[0]
        result.append({
            'story_id': one_row.story_id,
            'story_text': one_row.story_text,
            'questions': d_question.to_dict('records')
        })

with open('output/newsqa_pred_demo.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
