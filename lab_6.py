from preprocess import Tokenizer
from math import log2
import timeit

class DocList:

    def __init__(self):
        self.docs_list = []
        self.highest_docno = -1
        self.total_length = 0
        self.term_cat_map = {}
        self.cat_doc_map = {}
        self.cat_size = {}

    
    def assign_docno(self):
        self.highest_docno += 1
        return self.highest_docno

    def add_to_cat_size(self, doc):
        if self.cat_size.get(doc.cat):
            self.cat_size.update({doc.cat : 1 + self.cat_size.get(doc.cat)})
        else:
            self.cat_size.update({doc.cat : 1})

    def append_cat_safe(self, filename, cat, tk):
        tk = Tokenizer()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                doc = tk.load_and_tokenize_memory(line)
                if doc != []:
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

    def append_doc(self, doc):
        self.add_to_cat_size(doc)
        self.add_to_term_cat_map(doc)
        self.add_to_cat_doc_map(doc)

class Doc:
    def __init__(self, terms, docno, cat):
        self.terms = terms
        self.docno = docno
        self.cat = cat

def mutual_information(term, cat, doclist):
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
    # notcats = list(cat_size.keys())
    # notcats.remove(cat)
    # counter = 0
    # for notcat in notcats:
    #     if cat_doc_map.get(notcat):
    #         docs = cat_doc_map.get(notcat)
    #         for doc in docs:
    #             if not term in doc.terms:
    #                 counter += 1
    # n00 = counter

    #print('n11: {}, n10: {}, n01: {}, n00: {}'.format(n11, n10, n01, n00))
    ep = 0.000000001
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
    return mi_list_trunc

start = timeit.default_timer()
tk = Tokenizer()
doclist = DocList()
doclist.append_cat_safe('corpus1.txt', 1, tk)
corp1 = timeit.default_timer()
doclist.append_cat_safe('corpus2.txt', 2, tk)
corp2 = timeit.default_timer()
print('mutual information for corpus 1: {}'.format(mi_for_all_terms(doclist, 1)))
print('mutual information for corpus 2: {}'.format(mi_for_all_terms(doclist, 2)))
mi = timeit.default_timer()

print('runtime:\nprocess corpus 1: {}\nprocess corpus 2: {}\ncalculate mutual information: {}'.format(
    (corp1-start), (corp2-corp1), (mi-corp2)
))
