import pandas as pd
import numpy as np
from math import log2
from scipy import stats
from preprocess import Tokenizer
from math import log2
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import sklearn
import re
import string
import scipy
import numpy as np

from sklearn.metrics import classification_report
from math import floor, log2, log10
import random
from string import punctuation
from nltk.stem import PorterStemmer

ps = PorterStemmer()
ep = 0.000000000000001
metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']

# Tokenizer class used throughout this courswork
class Tokenizer:
    def __init__(self, limit=None):
        self.data = ''
        self.terms = []
        f = open('stopwords.txt', 'r')
        self.stops = f.readlines()
        self.stops = [stop.strip() for stop in self.stops]
        f.close()

    def tokenize(self):
        data = self.data.splitlines()
        self.data = ''
        words = []
        for line in data:
            line = line.replace('-', ' ')
            line = line.replace("\'", "'")
            list = line.split(" ")
            words = words + list

        data = words
        data = [word.strip(punctuation) for word in data]
        data = [word.lower() for word in data if word != ""]

        data = [ps.stem(word) for word in data if not(word in self.stops) and not('http' in word)]
        
        #data = [ps.stem(word) for word in data]

        data = [term.translate(str.maketrans("", "", punctuation)) for term in data]
        data = [word for word in data if word.isalpha()]

        self.terms = data
        
    def tokenize_sentiment(self):
        data = self.data.splitlines()
        self.data = ''
        words = []
        for line in data:
            #line = line.replace('-', ' ')
            line = line.replace("\'", "'")
            list = line.split(" ")
            words = words + list
        data = words
        data = [word.strip(punctuation) for word in data]
        data = [word.lower() for word in data if word != ""]

        data = [ps.stem(word) for word in data if not('http' in word)]
        
        #data = [ps.stem(word) for word in data]

        data = [term.translate(str.maketrans("", "", punctuation)) for term in data]
        data = [word for word in data if word.isalpha()]

        self.terms = data

    def load_file_disk(self, filename):
        f = open(filename, 'r', encoding='UTF-8')
        self.data = f.read()
        f.close()
        
    def load_file_memory(self, data):
        self.data = data

    def clear(self):
        self.terms = []
        self.data = ''

    def load_and_tokenize_file(self, filename):
        self.load_file_disk(filename)
        self.tokenize()
        processed_terms = self.terms
        self.clear()
        return processed_terms

    def load_and_tokenize_memory(self, data, special=None):
        self.load_file_memory(data)
        if special:
            self.tokenize_sentiment()
        else:
            self.tokenize()
        processed_terms = self.terms
        self.clear()
        return processed_terms

    def split_to_docs_on_line(self, filename):
        f = open(filename, 'r', encoding='UTF-8')
        if self.limit:
            corp = f.read(self.limit)
        else:
            corp = f.read()
        corp_list = corp.split('\n')
        #print(corp_list)
        corp_docs = [self.load_and_tokenise_memory(doc) for doc in corp_list]
        corp_docs = [doc for doc in corp_docs if doc != [] ]
        #print(corp_docs)
        f.close()
        return corp_docs

# ================================================================
#               the following modules were mostly
#               used for part 1: IR EVALUATION
# ===============================================================

def get_relevance_rank_actual(query, system, k, qrels_df, results_df):
    rels_for_q = qrels_df[qrels_df['query_id'] == query]
    output_for_q = results_df[(results_df['query_number'] == query) & (results_df['system_number'] == system)]

    # print('query {}\nsystem {}'.format(query, system))
    # print('relevant_for_query\n' + str(rels_for_q))
    # print('system_output\n' + str(output_for_q))

    relevance_rank = []
    for i in range(k):
        result_doc = output_for_q.iloc[i]
        relevant_doc = rels_for_q[rels_for_q['doc_id'] == result_doc['doc_number']]  

        #relevant doc may be empty
        if relevant_doc.shape[0] > 0:
            relevance = relevant_doc.iloc[0]['relevance']
            doc_rank = result_doc['rank_of_doc']
            relevance_rank.append((relevance, doc_rank))
    relevance_rank.sort(key=lambda x: x[1])
    return relevance_rank

# is the same for every system
def get_relevance_rank_ideal(query, k, qrels_df):
    relevance_rank = []
    rels_for_q = qrels_df[qrels_df['query_id'] == query]
    
    num_rels = rels_for_q.shape[0]
    # ensures we don't try to index df when k > num_rels, avoid index oob
    if num_rels < k:
        k = num_rels
    for i in range(k):
        relevant_doc = rels_for_q.iloc[i]
        rank = i+1 # rank starts at 1, index starts at 0
        relevance = relevant_doc['relevance']
        relevance_rank.append((relevance, rank))
    return relevance_rank 
        

def calculate_dcg_for_rr(relevance_rank):
    dcg = 0
    if relevance_rank:
        dcg = relevance_rank[0][0]
    for i in range(1,len(relevance_rank)):
        # where a doc has relevance 0 the corresponding term would be 0 
        # so relevance rank just ignores it
        next_term = (relevance_rank[i][0]/log2(relevance_rank[i][1]))
        dcg += next_term
    return dcg

def get_ideal_dcg(query, k, qrels_df):
    relevance_rank_ideal = get_relevance_rank_ideal(query, k, qrels_df)
    ideal_dcg = calculate_dcg_for_rr(relevance_rank_ideal)
    return ideal_dcg
    

def get_dcg(query, system, k, qrels_df, results_df):
    relevance_rank_actual = get_relevance_rank_actual(query, system, k, qrels_df, results_df)
    dcg_at_k = calculate_dcg_for_rr(relevance_rank_actual)
    return dcg_at_k
    
def norm_dcg(query, ideal_dcg, system, k, qrels_df, results_df):
    dcg = get_dcg(query, system, k, qrels_df, results_df)
    norm_dcg = dcg/ideal_dcg
    return norm_dcg


def confusion_matrix(qrels_df, results_df, k, query, system):
    rel_ids= qrels_df[qrels_df['query_id'] == query]['doc_id'].values[:k]
    results = results_df[(results_df['query_number'] == query) & (results_df['system_number'] == system)]['doc_number'].values[:k]
    tp = 0
    fp = 0
    fn = 0
    for result in results:
        if result in rel_ids:
            tp += 1
        else: 
            fp += 1
    
    for rel in rel_ids:
        if rel not in results:
            fn += 1
    return tp, fp, fn

def precision(qrels_df, results_df, k, query, system):
    tp, fp, fn = confusion_matrix(qrels_df, results_df, k, query, system)
    precision = tp/(tp + fp)
    return precision

def r_precision(qrels_df, results_df, query, system):
    num_relevant = qrels_df[qrels_df['query_id'] == query].shape[0]
    tp, fp, fn = confusion_matrix(qrels_df, results_df, num_relevant, query, system)
    precision = tp/(tp + fp)
    return precision

def recall(qrels_df, results_df, k, query, system):
    tp, fp, fn = confusion_matrix(qrels_df, results_df, k, query, system)
    recall = tp/(tp+fn)
    return recall

def average_precision(qrels_df, results_df, query, system):
    relevant_docs = qrels_df[qrels_df['query_id'] == query]['doc_id'].values
    retrieved_docs = results_df[(results_df['system_number'] == system) & (results_df['query_number'] == query)]['doc_number'].values
    average_precision = 0
    for k,retrieved in enumerate(retrieved_docs,1):
        if retrieved in relevant_docs:
            average_precision += precision(qrels_df, results_df, k, query, system)
    average_precision = average_precision/len(relevant_docs)
    return average_precision 


def generate_eval_dict(num_systems, num_queries, qrels_df, results_df):
    eval_dict = {}
    for q in range(1, num_queries+1):
        eval_dict.update({q : {} })
        ideal_dcg_10 = get_ideal_dcg(q, 10, qrels_df)
        ideal_dcg_20 = get_ideal_dcg(q, 20, qrels_df)
        for s in range(1, num_systems+1):
            ave_precision = average_precision(qrels_df, results_df, q, s)
            precision_10 = precision(qrels_df, results_df, 10, q, s)
            recall_50 = recall(qrels_df, results_df, 50, q, s)
            r_prec = r_precision(qrels_df, results_df, q, s)
            norm_10 = norm_dcg(q, ideal_dcg_10, s, 10, qrels_df, results_df)
            norm_20 = norm_dcg(q, ideal_dcg_20, s, 20, qrels_df, results_df)
            eval_dict.get(q).update({s : {'nDCG@10' : norm_10, 'nDCG@20' : norm_20,
             'P@10' : precision_10, 'R@50' : recall_50, 'r-precision' : r_prec, 'AP' : ave_precision}})
    
   
    return eval_dict

def generate_mean_dict(eval_dict, num_systems, num_queries):
    metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
    mean_dict = {}
    sdev_dict = {}
    for s in range(1,num_systems+1):
        mean_dict.update({s : {}})
        sdev_dict.update({s : {}})
        for q in range(1,num_queries+1):
            for metric in metrics:
                if not mean_dict.get(s).get(metric):
                    mean_dict.get(s).update({metric : [eval_dict.get(q).get(s).get(metric)]})
                else:
                    mean_dict.get(s)[metric].append(eval_dict.get(q).get(s).get(metric))
        for metric in metrics:
            scores = mean_dict.get(s)[metric]
            mean_dict.get(s)[metric] = np.mean(scores)
            sdev_dict.get(s).update({metric : np.std(scores)})
    return mean_dict, sdev_dict


            
def write_eval_to_file(eval_dict, mean_dict, num_systems, num_queries):
    metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
    f = open('ir_eval.csv', 'w')
    f.write('system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n')
    for s in range(1,num_systems+1):
        for q in range(1,num_queries+1):
            f.write('{},{}'.format(s, q))
            for metric in metrics:
                f.write(',{:.3f}'.format(eval_dict.get(q).get(s).get(metric)))
            f.write('\n')
            if q == num_queries:
                f.write('{},{}'.format(s, 'mean'))
                for metric in metrics:
                    f.write(',{:.3f}'.format(mean_dict.get(s).get(metric)))
                if s != num_systems:
                    f.write('\n')
    f.close()

        
def EVAL(qrel_filename, system_results_filename):  
    qrels = pd.read_csv(qrel_filename)
    results = pd.read_csv(system_results_filename)
    num_queries = len(qrels['query_id'].unique())
    num_systems = len(results['system_number'].unique())
    eval_dict = generate_eval_dict(num_systems, num_queries, qrels, results)
    mean_dict, sdev_dict = generate_mean_dict(eval_dict, num_systems, num_queries)
    write_eval_to_file(eval_dict, mean_dict, num_systems, num_queries)
    return eval_dict, sdev_dict, mean_dict, num_systems, num_queries

def get_top_two_dict(mean_dict, num_systems):
    top_two_dict = {}
    for metric in metrics:
        metric_means = []
        for s in range(1,num_systems+1):
            metric_means.append((s, mean_dict.get(s).get(metric)))
        metric_means.sort(key=lambda x: x[1], reverse=True)
        top_two_dict.update({metric : metric_means[:2]})
    return top_two_dict

def is_significant(mean_dict, sdev_dict, system1, system2, alpha, metric):
    zscore = (mean_dict.get(system2).get(metric) - mean_dict.get(system1).get(metric)) / sdev_dict.get(system1).get(metric)
    p_val = stats.norm.cdf(zscore)
    # print(mean_dict.get(system2).get(metric))
    # print(mean_dict.get(system1).get(metric))
    # print(p_val)
    sig = False
    if p_val < alpha:
        sig = True
    return sig

def significance_for_all(mean_dict, sdev_dict, num_systems):
    top_two_dict = get_top_two_dict(mean_dict, num_systems)
    for metric in metrics:
        top_two = top_two_dict.get(metric)
        top_system = top_two[0][0]
        second_system = top_two[1][0]
        sig = is_significant(mean_dict, sdev_dict, top_system, second_system, 0.0025, metric)
        place = ' NOT '
        if sig:
            place = ' '
            
        print('top system for {met}: {sys1}\nsecond system for {met}: {sys2}\n'
              'top system is{place}significantly better than second\n\n'.format(met=metric, sys1=top_system, sys2=second_system, place=place))


def ir_eval(qrel_filename, results_filename):
    eval_dict, sdev_dict, mean_dict, num_systems, num_queries = EVAL(qrel_filename,results_filename)
    significance_for_all(mean_dict, sdev_dict, num_systems)


# ================================================================
#               the following modules were mostly
#               used for part 2: TEXT ANALYSIS
# ===============================================================


class DocList:

    def __init__(self):
        self.docs_list = []
        self.docs_map = {}
        self.highest_docno = -1
        self.total_length = 0
        self.term_cat_map = {}
        self.cat_doc_map = {}
        self.cat_size = {}
        self.cat_list = []

    
    def assign_docno(self):
        self.highest_docno += 1
        return self.highest_docno

    def add_to_cat_size(self, doc):
        if self.cat_size.get(doc.cat):
            self.cat_size.update({doc.cat : 1 + self.cat_size.get(doc.cat)})
        else:
            self.cat_size.update({doc.cat : 1})

    def append_cat_safe(self, filename, cat, tk, limit=None):
        if cat not in self.cat_list:
            self.cat_list.append(cat)
        with open(filename, 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                doc = tk.load_and_tokenize_memory(line)
                if doc != []:
                    self.append_doc(Doc(doc, self.assign_docno(), cat))
                i += 1
                if limit:
                    if i >= limit:
                        break
                    
    def append_cat_safe_memory_processed(self, docs, cat, tk):
            if cat not in self.cat_list:
                self.cat_list.append(cat)   
                
            for doc in docs:
                self.append_doc(Doc(doc, self.assign_docno(), cat))
                
                    
                    
    def add_to_cat_doc_map(self, doc):
        if self.cat_doc_map.get(doc.cat):
            self.cat_doc_map.get(doc.cat).append(doc)
        else:
            self.cat_doc_map.update({doc.cat : [doc]})
        
    def add_to_term_cat_map(self, doc):
        for term in doc.terms:  
            if self.term_cat_map.get(term):
                if (doc.docno, doc.cat) not in self.term_cat_map[term]:
                    self.term_cat_map[term].append((doc.docno, doc.cat))
            else:
                self.term_cat_map[term] = [(doc.docno, doc.cat)]

    def add_to_docs_map(self, doc):
        if self.docs_map.get(doc.cat):
            self.docs_map.get(doc.cat).append(doc.terms)
        else:
            self.docs_map.update({doc.cat : [doc.terms]})

    def append_doc(self, doc):
        self.add_to_docs_map(doc)
        self.add_to_cat_size(doc)
        self.add_to_term_cat_map(doc)

class Doc:
    def __init__(self, terms, docno, cat):
        self.terms = terms
        self.docno = docno
        self.cat = cat

def doc_class_frequencies(term, cat, doclist):
    N = doclist.highest_docno
    term_cat_map = doclist.term_cat_map
    #cat_doc_map = doclist.cat_doc_map
    cat_size = doclist.cat_size
    
    pairs = term_cat_map.get(term)
    counter = 0
    for pair in pairs:
        if pair[1] == cat:
            counter += 1
    n11 = counter
    n01 = cat_size.get(cat) - n11
    n10 = len(pairs) - n11
    n00 = N - n11 - n10 - n01

    return n11, n01, n10, n00

def chi_squared(term, cat, doclist):
    n11, n01, n10, n00 = doc_class_frequencies(term, cat, doclist)
    num = (n11+n10+n01+n00)*(n11*n00 - n10*n01)**2
    denom = (n11+n01)*(n11+n10)*(n10+n00)*(n01+n00)
    return num/(denom+ep)


def mutual_information(term, cat, doclist):
    n11, n01, n10, n00 = doc_class_frequencies(term, cat, doclist)
    N = doclist.highest_docno

    first = n11/N * log2((N*n11+ep)/((n11+n10)*(n11+n01)+ep))
    second = n01/N * log2((N*n01+ep)/((n01+n00)*(n11+n01)+ep))
    third = n10/N * log2((N*n10+ep)/((n10+n11)*(n10+n00)+ep))
    fourth = n00/N * log2((N*n00+ep)/((n01+n00)*(n10+n00)+ep))
   
    return first + second + third + fourth

def mi_for_all_terms(doclist, cat):
    term_cat_map = doclist.term_cat_map
    mi_list = []
    for term in term_cat_map.keys():
        if term_cat_map.get(term):
            if len(term_cat_map.get(term)) >= 10:
                mi_list.append((term, mutual_information(term, cat, doclist)))
    mi_list.sort(key=lambda x: x[1], reverse=True)
    mi_list_trunc = mi_list[:10]
    return mi_list

def cs_for_all_terms(doclist, cat):
    term_cat_map = doclist.term_cat_map
    cs_list = []
    for term in term_cat_map.keys():
        if term_cat_map.get(term):
            if len(term_cat_map.get(term)) >= 10:
                cs_list.append((term, chi_squared(term, cat, doclist)))
    cs_list.sort(key=lambda x: x[1], reverse=True)
    cs_list_trunc = cs_list[:10]
    return cs_list

def create_common_dictionary(doclist):
    common_text = []
    for cat in doclist.cat_list:
        common_text = common_text + doclist.docs_map.get(cat)

    return Dictionary(common_text)


def train_lda(doclist, common_dictionary):
    common_corpus = []
    for cat in doclist.cat_list:
        common_corpus = common_corpus + [common_dictionary.doc2bow(text) for text in doclist.docs_map.get(cat)]
    lda = LdaModel(common_corpus, num_topics=20, id2word=common_dictionary)
    return lda

def get_overall_topic_probs_for_cat(common_dictionary, doclist, lda, cat):
    topic_prob_list = []
    topic_map = {}
    print('starting cat {}'.format(cat))
    for doc in doclist.docs_map.get(cat):
        scores = lda.get_document_topics(common_dictionary.doc2bow(doc))
        for pair in scores:
            if topic_map.get(pair[0]):
                topic_map.update({pair[0] : topic_map[pair[0]] + pair[1]})
            else: 
                topic_map.update({pair[0] : pair[1]})

    for topic in topic_map.keys():
        topic_prob_list.append((topic, topic_map.get(topic)/doclist.cat_size.get(cat)))
    
    topic_prob_list.sort(key=lambda x: x[1], reverse=True)
    tpl_trunc = topic_prob_list[:3]
    print('cat: {}, topic prob scores: {}'.format(cat, tpl_trunc))
    #topic_list = [pair[0] for pair in tpl_trunc]

    return tpl_trunc


def get_topic_probs_for_all_cat(common_dictionary, doclist, lda):
    topic_prob_list = []
    for cat in doclist.cat_list:
        topic_prob_list.append((cat, get_overall_topic_probs_for_cat(common_dictionary, doclist, lda, cat)))
    return topic_prob_list

def generate_cat_topic_words(lda, topic_prob_list, common_dictionary):
    cat_topic_words = []
    for pair in topic_prob_list:
        topic_ids = pair[1]
        for topic_id in topic_ids:
            cat_topic_words.append((pair[0], lda.print_topic(topic_id[0])))
    return cat_topic_words

def find_cat_topic_words_from_corpus(doclist):
    common_dictionary = create_common_dictionary(doclist)
    lda = train_lda(doclist, common_dictionary)
    topic_prob_list = get_topic_probs_for_all_cat(common_dictionary, doclist, lda)
    print(generate_cat_topic_words(lda, topic_prob_list, common_dictionary))



def text_anal(filename):
    doclist = DocList()
    tk = Tokenizer()
    f = open(filename, 'r')
    cat2id = {}
    id_counter = 1
    while True:
        line = f.readline()
        if not line:
            break
        pair = line.split('\t')
        verse = pair[1]
        cat = pair[0]
        if not cat2id.get(cat):
            cat2id.update({cat : id_counter})
            id_counter += 1
        verse_tokens = tk.load_and_tokenize_memory(verse)
        doclist.append_cat_safe_memory_processed([verse_tokens], cat2id.get(cat), tk)
    find_cat_topic_words_from_corpus(doclist)

    make_table(cat2id, doclist)

def make_table(cat2id, doclist):
    file = open('text_analysis.csv','w')
    file.write('Method\Rank')
    for i in range(1, 11):
        file.write(',{}'.format(i))
    for corpus in cat2id.keys():
        write_cs(file, corpus, cat2id, doclist)
        write_mi(file, corpus, cat2id, doclist)

def write_cs(file, corpus, cat2id, doclist):
    cs_list = cs_for_all_terms(doclist, cat2id.get(corpus))[:10]
    file.write('\nX^2_{}'.format(corpus))
    for cs in cs_list:
        file.write(',{}'.format(cs[0]))

def write_mi(file, corpus, cat2id, doclist):
    mi_list = mi_for_all_terms(doclist, cat2id.get(corpus))[:10]
    file.write('\nMI_{}'.format(corpus))
    for mi in mi_list:
        file.write(',{}'.format(mi[0]))


# ================================================================
#               the following modules were mostly
#               used for part 3: TEXT CLASSIFICATION
# ===============================================================



train_rat = 0.8 # ratio of train_dev
dev_rat = 0.2
# hyperparameters, they are explained in the report
num_of_mi_words = 2500
mi_mult = 15
all_caps_multiplier = 1

chars_to_remove = re.compile(f'[{string.punctuation}]')

def split_data(filename_train_dev):
    f = open(filename_train_dev,encoding="latin-1")
    data = f.readlines() 
    f.close()
    del data[0]
    random.shuffle(data)
    size = len(data)
    dev_documents = []
    dev_categories = []
    train_documents = []
    train_categories = []
    vocab = set([])
    train_threshold = size*train_rat
        
    for count,line in enumerate(data):
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            # split on tabs, we have 3 columns in this tsv format file
            tweet_id, category, tweet = line.split('\t')

            # process the words

            words = tweet.split()
            #words = chars_to_remove.sub('',tweet).lower().split()

            if count < train_threshold:
                for word in words:
                    vocab.add(word)
                # add the list of words to the documents list
                train_documents.append(words)
                # add the category to the categories list
                train_categories.append(category)

            else:
                dev_documents.append(words)
                dev_categories.append(category)
        
            
    return train_documents, dev_documents, train_categories, dev_categories, vocab

def process_test_data(filename_test):
    f = open(filename_test, encoding="latin-1")
    data = f.readlines()
    f.close()
    del data[0]
    test_documents = []
    test_categories = []
    for count,line in enumerate(data):
        line = line.strip().strip()
        if line:
            tweet_id, category, tweet = line.split('\t')
            words = tweet.split()
            test_documents.append(words)
            test_categories.append(category)

    return test_documents, test_categories


def make_word2id(vocab):
    word2id = {}
    for word_id,word in enumerate(vocab):
        word2id[word] = word_id
    return word2id

def make_id2df(docs, word2id):
    id2df = {}
    for doc in docs:
        for word in doc:
            if id2df.get(word2id.get(word)):
                id2df[word2id[word]] += 1
            else:
                id2df.update({word2id[word] : 1})
    return id2df

    
# and do the same for the categories
def make_cat2id_vv(cats):
    cat2id = {}
    id2cat = {}
    for cat_id,cat in enumerate(set(cats)):
        cat2id[cat] = cat_id
        id2cat[cat_id] = cat
    return cat2id, id2cat

def convert_to_bow_matrix_baseline(preprocessed_data, word2id):

    preprocessed_data = [[chars_to_remove.sub('',word).lower() for word in doc] for doc in preprocessed_data]
    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data),len(word2id)+1)
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    # iterate through all documents in the dataset
    for doc_id,doc in enumerate(preprocessed_data):
        for word in doc:
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id,word2id.get(word,oov_index)] += 1

    return X

def extra_processing(docs, cats):
    new_docs = []
    new_cats = []
    new_vocab = set([])
    tk = Tokenizer()
    for count,doc in enumerate(docs):
        new_doc = []
        for word in doc:
            if word.isupper():
                new_doc.append(chars_to_remove.sub('',word))
                new_vocab.add(chars_to_remove.sub('',word))
                
                
            word = word.lower()
            tkword = tk.load_and_tokenize_memory(data=word, special=True)
                           
            
            # if all_caps:
            #     for _ in range(all_caps_multiplier):
            #         for term in tkword:
            #             new_doc.append(term)
            #             new_vocab.add(term)
                
            if word[len(word)-1] == '!':
                new_doc.append('!')
                new_vocab.add('!')
            if word[0] == '#':
                new_doc.append(word)
                new_vocab.add(word)
                post_tag = tk.load_and_tokenize_memory(data=word[1:], special=True)
                for word in post_tag:
                    new_doc.append(word)
                    new_vocab.add(word)
                    
            elif word[0] == '@':
                new_doc.append(word)
                new_vocab.add(word)
                
                
            
            elif tkword:
                for word in tkword:
                    new_doc.append(word)
                    new_vocab.add(word)
                    
                    
        if new_doc:
            new_docs.append(new_doc)
            new_cats.append(cats[count])
    return new_docs, new_cats, new_vocab       
            

def convert_to_matrix_improve(data, word2id, id2df, mis):  
    matrix_size = (len(data), len(word2id)+2)
    oov_index = len(word2id)
    oov_value = 1 ## think about this value - I think I just want it pretty small
    tweet_len_index = len(word2id)+1
    X = scipy.sparse.dok_matrix(matrix_size)
    for doc_id,doc in enumerate(data):
        doc_len_feature = log2(len(doc))
        X[doc_id,tweet_len_index] = doc_len_feature
        for word in doc:
            word_id = word2id.get(word,oov_index)
            multiplier = 1
            if word_id in mis:
                multiplier = mi_mult
            if word_id == oov_index:
                X[doc_id,word_id] += oov_value
            else:
                #X[doc_id,word_id] += len(data)/id2df.get(word_id)
                X[doc_id,word_id] += 1 * multiplier
    return X


def create_eval_dict_per_set(true, preds, cat_names):
    return classification_report(true, preds, target_names=cat_names, output_dict=True)


def convert_to_cat_vector(categories, cat2id):
    cat_vector = []
    for cat in categories:
        cat_vector.append(cat2id.get(cat))
    return cat_vector


def generate_fitted_model(X_train, y_train, C):
    model = sklearn.svm.LinearSVC(C=C)
    model.fit(X_train, y_train)
    return model

def get_top_mi_words(new_docs, new_cats, word2id, cat2id, num):
        tk = Tokenizer()
        doclist = DocList()
    
        for i,doc in enumerate(new_docs):
            doclist.append_cat_safe_memory_processed([doc], cat2id.get(new_cats[i]), tk)
            
        mis = set([])
        for catid in cat2id.items():
            mi4all = mi_for_all_terms(doclist, catid[1])[:num]
            for mi in mi4all:
                mis.add(word2id.get(mi[0]))
        
        return mis


def make_X_train_and_fit_imp(docs, cats, C):
    # extra processing on documents
    new_docs, new_cats, vocab = extra_processing(docs, cats)
    word2id = make_word2id(vocab)
    cat2id, id2cat = make_cat2id_vv(new_cats)
    id2df = make_id2df(new_docs, word2id)
    
    mis = get_top_mi_words(new_docs, new_cats, word2id, cat2id, num_of_mi_words)
  
    X_train = convert_to_matrix_improve(new_docs, word2id, id2df, mis)
    y_train = convert_to_cat_vector(new_cats, cat2id)
    model = generate_fitted_model(X_train, y_train, C)
    return model, X_train, word2id, id2df, cat2id, id2cat, mis # also returns X_train bc need to predict on it


def make_X_train_and_fit_base(docs, cats, vocab, C):
    word2id = make_word2id(vocab)
    cat2id, _ = make_cat2id_vv(cats)
    X_train = convert_to_bow_matrix_baseline(docs, word2id)
    y_train = convert_to_cat_vector(cats, cat2id)
    model = generate_fitted_model(X_train, y_train, C)
    return model, X_train, word2id, cat2id


def get_preds_and_true_from_data_base(docs, cats, model, word2id, cat2id, id2cat):
    X = convert_to_bow_matrix_baseline(docs, word2id)
    y_true = convert_to_cat_vector(cats, cat2id)
    y_pred = model.predict(X)
    write_crime_scenes(docs, X, y_true, y_pred, id2cat)
    return y_true, y_pred

def get_preds_and_true_from_data_imp(docs, cats, model, word2id, id2df, cat2id, id2cat, mis):
    new_docs, new_cats, _ = extra_processing(docs, cats)
    X = convert_to_matrix_improve(new_docs, word2id, id2df, mis)
    y_true = convert_to_cat_vector(new_cats, cat2id)
    y_pred = model.predict(X)
    #write_crime_scenes(new_docs, X, y_true, y_pred, id2cat)
    return y_true, y_pred, new_docs


def make_cat_names(cat2id):
    cat_names = []
    for cat,cid in sorted(cat2id.items(),key=lambda x:x[1]):
        cat_names.append(cat)
    return cat_names

#writes to a text file every misclassification
def write_crime_scenes(docs, X, y_true, y_pred, id2cat):
    crime_scenes = []
    for i,pred in enumerate(y_pred):
        if y_true[i] != pred:
            #doc is only ever preprocessed. 
            #would take more bookkeeping to get actual orig doc. 
            
            crime_scenes.append((docs[i], X[i].items(), id2cat.get(pred), id2cat.get(y_true[i])))
    
    with open('crime_scenes.txt', 'w') as f:
        for crime in crime_scenes:
            f.write(str(crime) + '\n\n')
    return crime_scenes
    

def add_preds_to_file(file, true, preds, cat_names):
    dict = create_eval_dict_per_set(true, preds, cat_names)
    write_cat(dict, file, 'positive')
    write_cat(dict, file, 'negative')
    write_cat(dict, file, 'neutral')
    p_mac = dict.get('macro avg').get('precision')
    r_mac = dict.get('macro avg').get('recall')
    f_mac = dict.get('macro avg').get('f1-score')
    file.write('{},{},{}'.format(p_mac, r_mac, f_mac))

def write_cat(dict, file, cat):
    pos_dict = dict.get(cat)
    write_metrics(pos_dict, file)

def write_metrics(dict, file):
    precision = dict.get('precision')
    recall = dict.get('recall')
    f1 = dict.get('f1-score')
    file.write('{},{},{},'.format(precision, recall, f1))



def text_classification(filename_train_dev, filename_test):
    test_docs, test_cats = process_test_data(filename_test)
    train_docs, dev_docs, train_cats, dev_cats, vocab = split_data(filename_train_dev)

    base_model, X_train_base, word2id_base, cat2id_base = make_X_train_and_fit_base(train_docs, train_cats, vocab, 1000)
    imp_model, X_train_imp, word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp = make_X_train_and_fit_imp(train_docs, train_cats, 0.00005)

    cat_names_base = make_cat_names(cat2id_base)
    cat_names_imp = make_cat_names(cat2id_imp)

    y_true_train_base, y_pred_train_base = get_preds_and_true_from_data_base(train_docs, train_cats, base_model,
    word2id_base, cat2id_base, id2cat_imp)

    y_true_train_imp, y_pred_train_imp, docs_train_imp = get_preds_and_true_from_data_imp(train_docs, train_cats, imp_model,
    word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp)

    y_true_dev_base, y_pred_dev_base = get_preds_and_true_from_data_base(dev_docs, dev_cats, base_model,
    word2id_base, cat2id_base, id2cat_imp)

    y_true_dev_imp, y_pred_dev_imp, docs_dev_imp = get_preds_and_true_from_data_imp(dev_docs, dev_cats, imp_model,
    word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp)

    y_true_test_imp, y_pred_test_imp, docs_test_imp = get_preds_and_true_from_data_imp(test_docs, test_cats, imp_model,
    word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp)

    y_true_test_base, y_pred_test_base = get_preds_and_true_from_data_base(test_docs, test_cats, base_model,
    word2id_base, cat2id_base, id2cat_imp)

    f = open('classification.csv', 'w')
    f.write('system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro\n')
    f.write('baseline,train,')

    add_preds_to_file(f,y_true_train_base,y_pred_train_base, cat_names_base)
    f.write('\nbaseline,dev,')
    add_preds_to_file(f,y_true_dev_base,y_pred_dev_base, cat_names_base)
    f.write('\nbaseline,test,')
    add_preds_to_file(f,y_true_test_base,y_pred_test_base, cat_names_base)
    f.write('\nimproved,train,')
    add_preds_to_file(f,y_true_train_imp,y_pred_train_imp, cat_names_imp)
    f.write('\nimproved,dev,')
    add_preds_to_file(f,y_true_dev_imp,y_pred_dev_imp, cat_names_imp)
    f.write('\nimproved,test,')
    add_preds_to_file(f,y_true_test_imp,y_pred_test_imp, cat_names_imp)
    f.close()



ir_eval('qrels.csv', 'system_results.csv')
text_classification('tweets_train_dev.txt', 'tweets_test.txt')
text_anal('train_and_dev.tsv')