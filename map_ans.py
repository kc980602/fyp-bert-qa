import json
import pandas as pd


def append_ans(row):
    if row.is_answer_absent == 1 or row.validated_answers :
        ans_keys = json.loads(row.validated_answers)
        ans_list = []
        for ans_key in ans_keys:
            if ans_key not in ['none', 'bad_question']:
                ans_idx = list(map(int, ans_key.split(':')))
                ans_list.append(row.story_text[ans_idx[0]:ans_idx[1]])
        return ans_list


#   import source dataset
data = pd.read_csv('./data/newsqa_combined-newsqa-data-v1.csv', sep=',')
#   filter null
data = data.loc[data['validated_answers'].notnull()]
#   append ans
data['answers'] = data.apply(append_ans, axis=1)
# #   filter no answer
# data = data[data.answers.str.len() > 0]
# #   output
# data.to_csv('data/newsqa_w_ans.csv')