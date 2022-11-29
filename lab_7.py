from cmath import log10
import sklearn
import collections
import re
import string
import scipy
from scipy import sparse
import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import classification_report
import sys
from math import floor, log2
from preprocess import Tokenizer
import random

filename = sys.argv[1]
train_rat = 0.8
dev_rat = 0.2

chars_to_remove = re.compile(f'[{string.punctuation}]')

def split_data():
    f = open(filename,encoding="latin-1")
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

train_docs, dev_docs, train_cats, dev_cats, vocab = split_data()

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
            word = word.lower()
            tkword = tk.load_and_tokenize_memory(data=word)
            if word[0] == '#':
                new_doc.append(word)
                new_vocab.add(word)
                post_tag = tk.load_and_tokenize_memory(data=word[1:])
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
            
            


def convert_to_matrix_improve(data, word2id, id2df):  
    matrix_size = (len(data), len(word2id)+2)
    oov_index = len(word2id)
    oov_value = 10 ## think about this value - I think I just want it pretty small
    tweet_len_index = len(word2id)+1
    X = scipy.sparse.dok_matrix(matrix_size)
    for doc_id,doc in enumerate(data):
        X[doc_id,tweet_len_index] = len(doc)
        for word in doc:
            word_id = word2id.get(word,oov_index)
            # using log idf but linear tf because tweets are short but there are loads of them
            if word_id == oov_index:
                X[doc_id,word_id] += oov_value
            else:
                X[doc_id,word_id] += log2(len(train_docs)/id2df.get(word_id))
    return X


def convert_to_cat_vector(categories, cat2id):
    cat_vector = []
    for cat in categories:
        cat_vector.append(cat2id.get(cat))
    return cat_vector

train_docs_imp, train_cats_imp, vocab_imp = extra_processing(train_docs, train_cats)
dev_docs_imp, dev_cats_imp, _ = extra_processing(dev_docs, dev_cats)


word2id_imp = make_word2id(vocab_imp)
cat2id_imp, id2cat_imp = make_cat2id_vv(train_cats_imp)

word2id_base = make_word2id(vocab)
cat2id_base, id2cat_base = make_cat2id_vv(train_cats)


id2df_imp = make_id2df(train_docs_imp, word2id_imp)

X_train_base = convert_to_bow_matrix_baseline(train_docs, word2id_base)
X_train_imp = convert_to_matrix_improve(train_docs_imp, word2id_imp, id2df_imp)

y_true_train_base = convert_to_cat_vector(train_cats, cat2id_base)
y_true_train_imp = convert_to_cat_vector(train_cats_imp, cat2id_imp)
#X_test_base = convert_to_bow_matrix_baseline(test_docs, word2id)
#y_true_test = convert_to_cat_vector(test_cats, cat2id)

X_dev_base = convert_to_bow_matrix_baseline(dev_docs, word2id_base)
X_dev_imp = convert_to_matrix_improve(dev_docs, word2id_imp, id2df_imp)

y_true_dev = convert_to_cat_vector(dev_cats, cat2id_imp)


cat_names = []
for cat,cid in sorted(cat2id_imp.items(),key=lambda x:x[1]):
    cat_names.append(cat)


def create_eval_dict_per_set(true, preds):
    return classification_report(true, preds, target_names=cat_names, output_dict=True)

def generate_fitted_model(X_train, y_train, C):
    model = sklearn.svm.LinearSVC(C=C)
    model.fit(X_train, y_train)
    return model

base_model = generate_fitted_model(X_train_base, y_true_train_base, 1000)
base_train_preds = base_model.predict(X_train_base)
#base_test_preds = base_model.predict(X_dev_base) ##MUST CHANGE TEST WHEN TEST COMES
base_dev_preds = base_model.predict(X_dev_base)

imp_model = generate_fitted_model(X_train_imp, y_true_train_imp, 1000)

# X_train might be a different size, I might have compressed features only using big MI score words etc.
imp_train_preds = imp_model.predict(X_train_imp)
#imp_test_preds = imp_model.predict(X_dev_imp) ##MUST CHANGE TEST WHEN TEST COMES
imp_dev_preds = imp_model.predict(X_dev_imp)

def add_preds_to_file(file, true, preds):
    dict = create_eval_dict_per_set(true, preds)
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


f = open('classification.csv', 'w')
f.write('system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro\n')
f.write('baseline,train,')

add_preds_to_file(f,y_true_train_base,base_train_preds)
f.write('\nbaseline,dev,')
add_preds_to_file(f,y_true_dev,base_dev_preds)
f.write('\nbaseline,test,')
#add_preds_to_file(f,y_true_test,base_test_preds)
f.write('\nimproved,train,')
add_preds_to_file(f,y_true_train_imp,imp_train_preds)
f.write('\nimproved,dev,')
add_preds_to_file(f,y_true_dev,imp_dev_preds)
f.write('\nimproved,test,')
#add_preds_to_file(f,y_true_test,imp_test_preds)


#rest of writing code goes in here

#                                  #

f.close()
    

# print(classification_report(y_test, y_test_predictions, target_names=cat_names))


# pos_sample = ['cant', 'wait', 'to', 'see', 'ed', 'shearans', 'nips', 'live', 'on', 'stage', 'pure', 'buzzin']
# neg_sample = ['fucking', 'annoying', 'how', 'tiny', 'shearans', 'nips', 'were', 'last', 'night']
# neu_sample = ['i', 'dont', 'know', 'what', 'to', 'do', 'today']

# def tweet_to_sparse(tweet):
#     sample_tweet = scipy.sparse.dok_matrix((1, len(word2id)+1))
#     for word in tweet:
#         sample_tweet[0, word2id.get(word,len(word2id))] += 1
#     return sample_tweet

# pos = tweet_to_sparse(pos_sample)
# neg = tweet_to_sparse(neg_sample)
# neu = tweet_to_sparse(neu_sample)

# model = sklearn.svm.LinearSVC(C=1000)
# model.fit(X_train, y_train)
# pred_pos = model.predict(pos)
# pred_neg = model.predict(neg)
# pred_neu = model.predict(neu)
# print('pos {}\nneg {}\nneu {}'.format(pred_pos, pred_neg, pred_neu))
# print(cat2id)


# y_test_predictions = model.predict(X_test)