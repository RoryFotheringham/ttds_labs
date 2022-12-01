
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
from math import floor, log2, log10
from preprocess import Tokenizer
import random
import lab_6

filename = sys.argv[1]
train_rat = 0.8
dev_rat = 0.2
num_of_mi_words = 2500
mi_mult = 15
all_caps_multiplier = 1

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
        doclist = lab_6.DocList()
    
        for i,doc in enumerate(new_docs):
            doclist.append_cat_safe_memory_processed([doc], cat2id.get(new_cats[i]), tk)
            
        mis = set([])
        for catid in cat2id.items():
            mi4all = lab_6.mi_for_all_terms(doclist, catid[1])[:num]
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

def get_preds_and_true_from_data_base(docs, cats, model, word2id, cat2id):
    X = convert_to_bow_matrix_baseline(docs, word2id)
    y_true = convert_to_cat_vector(cats, cat2id)
    y_pred = model.predict(X)
    return y_true, y_pred

def get_preds_and_true_from_data_imp(docs, cats, model, word2id, id2df, cat2id, id2cat, mis):
    new_docs, new_cats, _ = extra_processing(docs, cats)
    X = convert_to_matrix_improve(new_docs, word2id, id2df, mis)
    y_true = convert_to_cat_vector(new_cats, cat2id)
    y_pred = model.predict(X)
    write_crime_scenes(new_docs, X, y_true, y_pred, id2cat)
    return y_true, y_pred, new_docs


def make_cat_names(cat2id):
    cat_names = []
    for cat,cid in sorted(cat2id.items(),key=lambda x:x[1]):
        cat_names.append(cat)
    return cat_names

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
    
        


base_model, X_train_base, word2id_base, cat2id_base = make_X_train_and_fit_base(train_docs, train_cats, vocab, 1000)
imp_model, X_train_imp, word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp = make_X_train_and_fit_imp(train_docs, train_cats, 0.00005)

cat_names_base = make_cat_names(cat2id_base)
cat_names_imp = make_cat_names(cat2id_imp)

y_true_train_base, y_pred_train_base = get_preds_and_true_from_data_base(train_docs, train_cats, base_model,
 word2id_base, cat2id_base)

y_true_train_imp, y_pred_train_imp, docs_train_imp = get_preds_and_true_from_data_imp(train_docs, train_cats, imp_model,
 word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp)

y_true_dev_base, y_pred_dev_base = get_preds_and_true_from_data_base(dev_docs, dev_cats, base_model,
 word2id_base, cat2id_base)

y_true_dev_imp, y_pred_dev_imp, docs_dev_imp = get_preds_and_true_from_data_imp(dev_docs, dev_cats, imp_model,
 word2id_imp, id2df_imp, cat2id_imp, id2cat_imp, mis_imp)




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


f = open('classification.csv', 'w')
f.write('system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro\n')
f.write('baseline,train,')

add_preds_to_file(f,y_true_train_base,y_pred_train_base, cat_names_base)
f.write('\nbaseline,dev,')
add_preds_to_file(f,y_true_dev_base,y_pred_dev_base, cat_names_base)
f.write('\nbaseline,test,')
#add_preds_to_file(f,y_true_test,base_test_preds)
f.write('\nimproved,train,')
add_preds_to_file(f,y_true_train_imp,y_pred_train_imp, cat_names_imp)
f.write('\nimproved,dev,')
add_preds_to_file(f,y_true_dev_imp,y_pred_dev_imp, cat_names_imp)
f.write('\nimproved,test,')
#add_preds_to_file(f,y_true_test,imp_test_preds)


f.close()
    
