# import bertopic
# from bertopic import BERTopic
# from preprocess import preprocess
# from preprocess import preprocess_bertopic
# import pandas as pd
#
# # data = pd.read_csv('/Users/kirillsemenov/PycharmProjects/preprocess/venv/marina_updated_translated.csv').drop(
# #     columns='Unnamed: 0')
# # data = preprocess(data)
#
#
# def extract_topics(data):
#     print('extracting topics')
#     docs = preprocess_bertopic(data)
#     topic_model = BERTopic(nr_topics=30)
#     topics, probs = topic_model.fit_transform(docs)
#     # topic_model.get_topic_info()
#     data['topic_num'] = topics.astype(str)
#     topic_to_name = topic_model.get_topic_info()[['Topic', 'Name']].set_index('Topic')
#     return data
#
# if __name__ == '__main__':
#     data = pd.read_csv('/Users/kirillsemenov/PycharmProjects/preprocess/venv/marina_updated_translated.csv').drop(columns='Unnamed: 0')
#     data = preprocess(data)
#     data = extract_topics(data)