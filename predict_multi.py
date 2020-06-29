import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from tqdm.auto import tqdm
import pandas as pd


def predict(row):
    try:
        question, text = row.question, row.para_text
        if question != '' and pd.notnull(question) and text != '' and pd.notnull(text):
            encoding = tokenizer.encode_plus(question, text)
            input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
            start_scores, end_scores = model(tf.constant(input_ids)[None, :],
                                             token_type_ids=tf.constant(token_type_ids)[None, :])
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer = ' '.join(all_tokens[tf.math.argmax(tf.squeeze(start_scores)): tf.math.argmax(
                tf.squeeze(end_scores)) + 1]).replace(' ##', '')
        else:
            answer = ''
            print('Empty', row.story_id)
    except:
        print('Error:', row)
        answer = ''
    return answer


data = pd.read_csv('newsqa_w_para.csv', sep=',')
#   limit data size
# data_st, data_ed = 100000, 0
# data = data.iloc[data_st:]
# data = data.head(data_ed)
data.info()

chunk_size = int(data.shape[0] / 100)
chunks = [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]

try:
    with tf.device('/device:GPU:0'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        for idx, chunk in enumerate(chunks):
            tqdm.pandas()

            chunk['pred_ans'] = chunk.progress_apply(predict, axis=1)

            fname = 'pred_ans_{}.csv'.format(str(idx))
            chunk.to_csv('./prediction/{}'.format(fname))

except RuntimeError as e:
    print(e)
