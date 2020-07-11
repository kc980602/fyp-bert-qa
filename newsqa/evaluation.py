import string

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy
import re
import pandas as pd


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def similarity(answers):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(answers)
    cosim = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)
    return numpy.where(numpy.amax(cosim[0][1:]) > 0.5, 1, 0)


def get_answers(row):
    answers = []

    sourcers = row.answer_char_ranges.split('|')

    for source in sourcers:
        if source != 'None':
            ans_rngs = source.split(',')

            for rng in ans_rngs:
                ans_idx = list(map(int, rng.split(':')))
                answers.append(row.story_text[ans_idx[0]:ans_idx[1]])

    return answers


def classification(row):
    result = 0

    if row.pred_ans not in [numpy.nan, None]:
        answers = [row.pred_ans] + get_answers(row)
        answers = [normalize_answer(text) for text in answers]

        try:
            result = similarity(answers)
        except:
            #   extra checking for number/single char ans, TfidfVectorizer don't support this cases
            result = int(answers[0] in answers[1:])

        #   extra checking
        if result == 0:
            pred = answers[0]
            for text in answers[1:]:
                if pred.count(text) > 0 or text.count(pred) > 0 or re.sub('\W+', '', pred) == text or (
                        text[-1] == 's' and pred.count(text[0:-1]) > 0):
                    result = 1
                    break
    else:
        if len(get_answers(row)):
            result = 1

    return result


def exact_match(row):
    result = 0

    if row.pred_ans not in [numpy.nan, None]:
        answers = [row.pred_ans] + get_answers(row)
        answers = [normalize_answer(text) for text in answers]

        result = int(answers[0] in answers[1:])
    else:
        if len(get_answers(row)):
            result = 1
    return result


def correct_para(row):
    if pd.isnull(row.para_text):
        return 0
    try:
        text = re.sub(r'[^A-Za-z0-9\s]+', 'X', row.story_text)
        para = re.sub(r'[^A-Za-z0-9\s]+', 'X', row.para_text)
    except:
        print(row.story_text)
        print(row.para_text)

    para_start = para[0:15]
    para_end = para[-15:]

    pos_start, pos_end = '', ''

    for match in re.finditer(para_start, text):
        pos_start = match.start()

    for match in re.finditer(para_end, text):
        pos_end = match.end()

    if pos_start != '' and pos_end != '':
        sourcers = row.answer_char_ranges.split('|')

        for source in sourcers:
            if source != 'None':
                ans_rngs = source.split(',')
                for rng in ans_rngs:
                    ans_idx = list(map(int, rng.split(':')))
                    if pos_start <= ans_idx[0] and pos_end >= ans_idx[1]:
                        return 1

    return 0


version = '_v3'
data = pd.read_csv('../output/pred_ans{}.csv'.format(version), sep=',')
data = data.drop(columns=['Unnamed: 0'])
tqdm.pandas()
data['exact_match'] = data.progress_apply(exact_match, axis=1)
data['class'] = data.progress_apply(classification, axis=1)
data['is_correct_para'] = data.progress_apply(correct_para, axis=1)

data.info()

#   figures
total = data.shape[0]
correct = data[data['class'] == 1].shape[0]
returned = data[data['pred_ans'].notnull()].shape[0]
should_return = data[(data['is_question_bad'] == '1.0') | (data['is_answer_absent'] == 1)].shape[0]
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

data.to_csv('../output/pred_ans_w_eva{}.csv'.format(version))
