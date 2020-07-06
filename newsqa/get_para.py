from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd
import numpy as np


def find_best_match_para(para_list, question):
    train_set = [question] + para_list

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)
    cosim = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)

    cs_result = cosim[0][1:]
    matrix = np.where(cs_result == np.amax(cs_result))
    return matrix[0][0]


def split_para(text):
    para_list = [para for para in text.split('\n\n') if len(para)]
    merged = []
    tmp_para = ''
    for para in para_list:
        if len(tmp_para) + len(para) < 512:
            if len(tmp_para):
                sep = '\n\n'
            else:
                sep = ''
            tmp_para = tmp_para + sep + para
        else:
            merged.append(tmp_para)
            tmp_para = ''

    if len(tmp_para):
        merged.append(tmp_para)

    return merged


def get_para(row):
    question, text = row.question, row.story_text
    try:
        if len(text) >= 512:
            para_list = split_para(row.story_text)
            text_idx = find_best_match_para(para_list, question)
            text = para_list[text_idx]
    except:
        print('Error:', row)
    return text

data = pd.read_csv('../data/newsqa_combined-newsqa-data-v1.csv', sep=',')
#   filter empty question, assume is not null before using the QA bot
data = data.loc[data['question'].notnull()]
#   print data info
data.info()
#   add column
tqdm.pandas()
data['para_text'] = data.progress_apply(get_para, axis=1)
#   export result
data.to_csv('../output/newsqa_w_para.csv')
