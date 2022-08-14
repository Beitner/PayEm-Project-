import pandas as pd
import numpy as np
import time
import datetime
import datetime
from tqdm.notebook import tqdm
tqdm.pandas()
# import googletrans
# from googletrans import Translator
import bertopic
from bertopic import BERTopic
from time import mktime
from datetime import datetime
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
    #for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
    #for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
    # bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
    #for word embedding
import gensim
from gensim.models import Word2Vec
nltk.download('stopwords')

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
nltk.download('omw-1.4')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(mode='min',patience=5)



def day_time(data):
    """

    :param data: current data
    :return: data with 3 additional features created_time_mon, created_time_day, created_time_week.

    """
    print('change to daytime_index')

    def utc2local(row):
        utc = row['created']
        epoch = time.mktime(utc.timetuple())
        offset = datetime.fromtimestamp(epoch) - datetime.utcfromtimestamp(epoch)
        row['created_time'] = utc + offset

    data['created_time'] = data['created']
    data.apply(utc2local, axis=1)

    data['created_time_mon'] = data['created_time'].dt.month
    data['created_time_day'] = data['created_time'].dt.day
    data['created_time_week'] = data['created_time'].dt.week
    return data


def translate_text_features(data):
    """

    :param data: current data
    :return: data with text features translated to english

    """
    print('translating')
    data['request_reason'] = data['request_reason'].astype(str)
    data['title'] = data['title'].astype(str)

    def is_heb(str):
        alp_hebrew = "אבגדהוזחטיכלמנסעפצקרשתףץםך"
        text = str
        for char in alp_hebrew:
            if char in str:
                translator = Translator()
                translation = translator.translate(str, dest='en')
                text = translation.text
                break
        return (text)

    data['request_reason_Eng'] = data.request_reason.apply(lambda x: is_heb(x))
    data['title_Eng'] = data.title.apply(lambda x: is_heb(x))
    return data


def missing_vals(data):
    """

    :param data: current data
    :return: data with missing values filled in all features

    """
    print('filling the missing values')
    id_cols = ['purchase_request_id', '_accounting_id', '_department_id', '_sub_company_id', 'budget_item_id',
               'send_budget_to_user_id', 'vendor_id', ]

    for id_col in id_cols:
        data[id_col].fillna(value="UNKNOWN", inplace=True)
        data[id_col] = data[id_col].astype(str)
        data[id_col].fillna(value="UNKNOWN", inplace=True)

    data['title'].fillna(value='UNKNOWN', inplace=True)
    data['request_reason'].fillna(value='UNKNOWN', inplace=True)

    data['occurance'].fillna(value='UNKNOWN', inplace=True)

    return data


def status_to_binary(data):
    """

    :param data: current data
    :return: data with label 'status' changed to binary ('1'/'0') for ('APPROVED/DECLINED)

    """
    print('changing status to binary')
    NOT_PENDING_MASK = data.status != 'PENDING'
    data = data[NOT_PENDING_MASK]
    PA = data.status == 'PARTIALLY_APPROVED'
    data.loc[PA, 'status'] = 'APPROVED'
    OFFLINE_APPROVED = data.status == 'APPROVED_OFFLINE'
    data.loc[OFFLINE_APPROVED, 'status'] = 'APPROVED'

    data.loc[data['status'] == 'APPROVED', 'status'] = '1'
    data.loc[data['status'] == 'DECLINED', 'status'] = '0'
    return data


def currency(data):
    """

    :param data: current data
    :return: current data with additional feature 'currency'

    """
    print('extracting currency')
    pat = r'.+(\"currency\"): \"([a-zA-Z][a-zA-Z][a-zA-Z]).+'
    repl = lambda m: m.group(2)
    data.fx_info = data.fx_info.str.replace(pat, repl, regex=True, )
    data.fx_info.unique()
    mask2 = data.fx_info == '{}'
    data.loc[mask2, 'fx_info'] = 'UNKNOWN'
    data.rename(columns={"fx_info": "currency"}, inplace=True)
    return data


def to_categorical(data):
    """

    :param data: current data
    :return: data with some features changed to str type (made categorical)
    """
    print('changing to categorical')
    data['id'] = data['id'].astype(str)
    data['categories'] = data['categories'].astype(str)
    data['request_type'] = data['request_type'].astype(str)
    data['user_id'] = data['user_id'].astype(str)
    data.created_time_mon = data.created_time_mon.astype(str)
    data.created_time_day = data.created_time_day.astype(str)
    data.created_time_week = data.created_time_week.astype(str)
    return data


def drop_cols(data):
    """

    :param data: current data
    :return: data with redundant columns dropped

    """
    print('dropping redundant columns')
    data.drop(columns=['invoice', 'approved_occurance', 'decline_reason', 'declined_by_id', 'deleted'], inplace=True)
    data.drop(columns=['approved_amount', 'updated', 'used', 'polymorphic_ctype_id', 'updated'], inplace=True)
    return data


def preprocess_bertopic(data):
    """

    :param data: current data
    :return: list of strings preprocessed (removed stopwords/numbers), where each string is
    two text features concatenated for each of the data instance

    """

    print('preprocessing for bertopic')
    data['title_Eng'].fillna(value='UNKNOWN', inplace=True)
    data['request_reason_Eng'].fillna(value='UNKNOWN', inplace=True)
    data['title_reason'] = data['title_Eng'] + \
        ' - ' + data['request_reason_Eng']
    docs = data['request_reason_Eng'][data['request_reason_Eng']
                                      != 'UNKNOWN - UNKNOWN'].to_list()

    def remove_stopwords(docs):
        english_stopwords = stopwords.words('english')
        english_stopwords.append('unknown')
        for i in range(len(docs)):
            tokens = word_tokenize(docs[i].lower())
            tokens_wo_stopwords = [
                t for t in tokens if t not in english_stopwords]
            docs[i] = " ".join(tokens_wo_stopwords)
        return docs

    def remove_nums(docs):
        for i in range(len(docs)):
            docs[i] = ''.join(j for j in docs[i] if (
                j.isalpha() or j.isspace()))
        return docs

    docs = remove_nums(docs)
    docs = remove_stopwords(docs)
    return docs


def extract_topics(data):
    """

    :param data: current data
    :return: dataframe with new categorical feature topic_num, indicating the topic of the
    request after running bertopic clustering / saves bertopic model, needed for clustering requests during inference

    """
    docs = preprocess_bertopic(data)
    print('extracting topics')
    topic_model = BERTopic(nr_topics=30)
    topics, probs = topic_model.fit_transform(docs)
    data['topic_num'] = topics
    data['topic_num'] = data['topic_num'].astype(str)
    # topic_to_name = topic_model.get_topic_info()[['Topic', 'Name']].set_index('Topic')
    # topic_model.save("bertopic_model")
    return data, topic_model


def get_topic(data, topic_model):
    """

    :param data: current data
    :param topic_model: bertopic model from training
    :return: dataframe with new categorical feature topic_num, indicating the topic of the request

    """
    docs = preprocess_bertopic(data)
    print('extracting topic')
    topics, probs = topic_model.transform(docs)
    data['topic_num'] = topics
    data['topic_num'] = data['topic_num'].astype(str)

    return data


def preprocess(data, translated=True, betropic_model=None):
    """

    :param data: raw data
    :return: preprocesses data - ready for training

    """

    data = day_time(data)

    data = missing_vals(data)

    if not translated:  # if train data is not yet translated then translate text features
        data = translate_text_features(data)

    data['invoice_attached'] = data['invoice'].notnull().astype(str)

    data = currency(data)

    data = status_to_binary(data)

    data = to_categorical(data)

    data = drop_cols(data)

    if betropic_model is None:  # for train stage
        data, topic_model = extract_topics(data)
        return data, topic_model
    else:  # for inference stage
        data = get_topic(data, betropic_model)

    return data


