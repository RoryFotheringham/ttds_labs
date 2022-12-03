from preprocess import Tokenizer
from math import log2
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import timeit
ep = 0.000000000000001
from numpy import mean

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


def text_classification():
    doclist = DocList()
    tk = Tokenizer()
    f = open('train_and_dev.tsv', 'r')
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

text_classification()




