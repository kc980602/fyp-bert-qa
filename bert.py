import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


def find_best_match_para(para_list, question):
    text_tokens = word_tokenize(question)
    tokens_without_sw = [word for word in text_tokens if not word.lower() in stopwords.words('english')]
    clear_query = re.sub(r'[^A-Za-z0-9\s]+', '', ' '.join(tokens_without_sw))

    train_set = [clear_query] + para_list

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)
    cosim = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)

    cs_result = cosim[0][1:]
    matrix = numpy.where(cs_result == numpy.amax(cs_result))
    return matrix[0][0]


class QA:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        print('QA init done')

def predict(self, para_list, question):
    text_id = find_best_match_para(para_list, question)
    text = para_list[text_id]
        encoding = self.tokenizer.encode_plus(question, text)
    encoding = self.tokenizer.encode_plus(question, text, max_length=512)
    input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
    start_scores, end_scores = self.model(tf.constant(input_ids)[None, :],
                                          token_type_ids=tf.constant(token_type_ids)[None, :])
    all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[tf.math.argmax(tf.squeeze(start_scores)): tf.math.argmax(tf.squeeze(end_scores)) + 1]).replace(' ##', '').replace('[CLS]', '').replace('[SEP]', '')

    ans_qus = [re.sub(r'[^A-Za-z0-9\s]+', '', text).strip().lower().replace(' ', '') for text in [answer, question]]
    if ans_qus[0] == ans_qus[1]:
        answer = ''

    return text_id, text, answer
