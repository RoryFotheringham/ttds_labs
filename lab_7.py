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
from math import floor

filename = sys.argv[1]
train_rat = 0.7
dev_rat = 0.1
test_rat = 0.2

f = open(filename,encoding="latin-1")
data = f.readlines() 
f.close()
size = len(data)
tickets = np.arange(0, size)
train_numbers = set(np.random.choice(tickets, size=floor(size*train_rat), replace=False))
dev_numbers = set(np.random.choice(tickets, size=floor(size*dev_rat), replace=False))
test_numbers = set(np.random.choice(tickets, size=floor(size*test_rat), replace=False))


def preprocess_data_baseline(data, train_numbers, dev_numbers, test_numbers):
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    dev_documents = []
    dev_categories = []
    test_categories = []
    test_documents = []
    train_documents = []
    train_categories = []
    vocab = set([])
    
    #lines = data.split('\n')
    
    for count,line in enumerate(data):
        if count == 0:
            continue
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            # split on tabs, we have 3 columns in this tsv format file
            tweet_id, category, tweet = line.split('\t')

            # process the words

            words = chars_to_remove.sub('',tweet).lower().split()

            if count in train_numbers:
                for word in words:
                    vocab.add(word)
                # add the list of words to the documents list
                train_documents.append(words)
                # add the category to the categories list
                train_categories.append(category)
            
            elif count in test_numbers:
                test_documents.append(words)
                test_categories.append(category)

            elif count in dev_numbers:
                dev_documents.append(words)
                dev_documents.append(category)
            
    return train_documents, test_documents, dev_documents, train_categories, test_categories, dev_categories, vocab




train_docs, test_docs, dev_docs, train_cats, test_cats, dev_cats, vocab = preprocess_data_baseline(data, train_numbers, test_numbers, dev_numbers)

word2id = {}
for word_id,word in enumerate(vocab):
    word2id[word] = word_id
    
# and do the same for the categories
cat2id = {}
id2cat = {}
for cat_id,cat in enumerate(set(train_cats)):
    cat2id[cat] = cat_id
    id2cat[cat_id] = cat

def convert_to_bow_matrix_baseline(preprocessed_data, word2id):

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

def convert_to_cat_vector(categories, cat2id):
    cat_vector = []
    for cat in categories:
        cat_vector.append(cat2id.get(cat))
    return cat_vector

X_train = convert_to_bow_matrix_baseline(train_docs, word2id)
y_train = convert_to_cat_vector(train_cats, cat2id)

X_test = convert_to_bow_matrix_baseline(test_docs, word2id)
y_test = convert_to_cat_vector(test_cats, cat2id)

X_dev = convert_to_bow_matrix_baseline(dev_docs, word2id)
y_dev = convert_to_cat_vector(dev_cats, cat2id)

pos_sample = ['cant', 'wait', 'to', 'see', 'ed', 'shearans', 'nips', 'live', 'on', 'stage', 'pure', 'buzzin']
neg_sample = ['fucking', 'annoying', 'how', 'tiny', 'shearans', 'nips', 'were', 'last', 'night']
neu_sample = ['i', 'dont', 'know', 'what', 'to', 'do', 'today']

def tweet_to_sparse(tweet):
    sample_tweet = scipy.sparse.dok_matrix((1, len(word2id)+1))
    for word in tweet:
        sample_tweet[0, word2id.get(word,len(word2id))] += 1
    return sample_tweet

pos = tweet_to_sparse(pos_sample)
neg = tweet_to_sparse(neg_sample)
neu = tweet_to_sparse(neu_sample)

model = sklearn.svm.LinearSVC(C=1000)
model.fit(X_train, y_train)
pred_pos = model.predict(pos)
pred_neg = model.predict(neg)
pred_neu = model.predict(neu)
print('pos {}\nneg {}\nneu {}'.format(pred_pos, pred_neg, pred_neu))
print(cat2id)


y_test_predictions = model.predict(X_test)
cat_names = []
for cat,cid in sorted(cat2id.items(),key=lambda x:x[1]):
    cat_names.append(cat)

print(classification_report(y_test, y_test_predictions, target_names=cat_names))


def predict_on_sets(model):
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)
    y_dev_preds = model.predict(X_dev)

    return y_train_preds, y_test_preds, y_dev_preds

def create_eval_dict_per_set(true, preds):
    return classification_report(true, preds, target_names=cat_names, output_dict=True)

f = open('classification.csv', 'w')
f.write('system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro')

def add_preds_to_file(file, true, preds):
    dict = create_eval_dict_per_set(true, preds)
    write_cat(dict, file, 'positive')
    write_cat(dict, file, 'negative')
    write_cat(dict, file, 'neutral')
    p-mac = dict.get('macro avg').get('precision')
    r-mac = dict.get('macro avg').get('recall')
    f-mac = dict.get('macro avg').get('f1-score')
    file.write('{},{},{}'.format(p-mac, r-mac, f-mac))

def write_cat(dict, file, cat):
    pos_dict = dict.get(cat)
    write_metrics(pos_dict, file)

def write_metrics(dict, file):
    precision = dict.get('precision')
    recall = dict.get('recall')
    f1 = dict.get('f1-score')
    file.write('{},{},{},'.format(precision, recall, f1))
    
create_eval_dict_per_set(y_test, y_test_predictions)


