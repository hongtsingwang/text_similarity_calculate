#!/usr/bin/env python
# coding=utf-8

import os
import sys
import jieba.posseg as pseg
import codecs
import logging
from gensim import corpora, models, similarities
reload(sys)
sys.setdefaultencoding("utf-8")

# pseg 带有词性标注的分词器 
logging.basicConfig(
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        level=logging.INFO,
        datefmt='%a, %d %b %Y %H:%M:%S'
                   )
home_dir = os.getcwd()
data_dir = os.path.join(home_dir, "data")
assert os.path.isdir(data_dir)

stop_words_file = os.path.join(data_dir, 'stop_words.txt')
stopwords = [x.strip() for x in codecs.open(stop_words_file, 'r', encoding='utf8').readlines()]
stop_flag = set(['c', 'd', 'f', 'm', 'p', 'r', 't', 'u', 'uj', 'x']) # 一些不可用的词性，一旦词语的词性在这其中，那么久过滤掉 按照次序， 分别为 连词，副词，方位词，数词，介词，代词，时间词，助词，过，字符串


def tokenization(file_name):
    result = []
    input_file = codecs.open(file_name, "r", encoding="utf-8")
    words = pseg.cut(input_file.read())
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    # logging.debug("result length is " % len(result))
    return result


file_names = [
             os.path.join(data_dir, '帮您减压的13件小事.txt'),
             os.path.join(data_dir, '高血压患者多喝脱脂牛奶.txt'),
             os.path.join(data_dir, 'ios.txt')
            ]

corpus = []
for file_name in file_names:
    corpus.append(tokenization(file_name))

logging.debug("the length of corpus:%s" % len(corpus))

dictionary = corpora.Dictionary(corpus)
# logging.info("the dictionary is %s" % ",".join(dictionary))

doc_vectors = [dictionary.doc2bow(text) for text in corpus]
logging.info("length of doc_vectors is %d" % len(doc_vectors))
# logging.debug("doc_vectors is %s" % ",".join(doc_vectors))


def sim_cal_tfidf(doc_vector=None, input_file=""):
    tfidf = models.TfidfModel(doc_vector)
    tfidf_vectors = tfidf[doc_vector]
    # logging.info("length of vector is :%d" % tfidf_vectors)
    # logging.info("vector[0] is %s" % ",".join(tfidf_vectors[0]))
    query = tokenization(input_file)
    query_bow = dictionary.doc2bow(query)
    # logging.info("query_bow is %s" % ",".join(query_bow))
    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_bow]
    return list(enumerate(sims)), tfidf_vectors


def sim_cal_lsi(doc_vector=None, input_file="", tfidf_vectors=None):
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=2)
    lsi.print_topics(2)
    lsi_vector = lsi[tfidf_vectors]
    for vec in lsi_vector:
        print vec
    query = tokenization(input_file)
    query_bow = dictionary.doc2bow(query)
    query_lsi = lsi[query_bow]
    index = similarities.MatrixSimilarity(lsi_vector)
    sims = index[query_lsi]
    return list(enumerate(sims))



simlarity_tfidf,tfidf_vectors = sim_cal_tfidf(doc_vectors, 'data/关于降压药的若干个问题.txt')
print "tfidf", simlarity_tfidf
simlarity_lsi = sim_cal_lsi(doc_vectors,'data/关于降压药的若干个问题.txt', tfidf_vectors)
print "lsi", simlarity_lsi
