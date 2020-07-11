import pandas as pd

data = pd.read_csv('../output/pred_ans_w_eva_v3.csv', sep=',')


#   figures
total = data.shape[0]
correct = data[data['class'] == 1].shape[0]
returned = data[data['pred_ans'].notnull()].shape[0]
should_return = data[(data['is_question_bad'] != '1.0') & (data['is_answer_absent'] != 1)].shape[0]
is_correct_para = data[data['is_correct_para'] == 1].shape[0]

precision = correct / returned
recall = correct / should_return
f1 = 2 * (precision * recall) / (precision + recall)
em = data[data['exact_match'] == 1].shape[0] / should_return

print(
    'Total: {}\nCorrect: {} [{}]\nReturned: {} [{}]\nShould Return: {} [{}]\nPrecision: {}\nRecall: {}\nF1 Score: {}\nEM Score: {}\nIs Correct Para: {} [{}]'
        .format(
        total,
        correct / total,
        correct,
        returned / total,
        returned,
        should_return / total,
        should_return,
        precision,
        recall,
        f1,
        em,
        is_correct_para / should_return,
        is_correct_para
    ))
