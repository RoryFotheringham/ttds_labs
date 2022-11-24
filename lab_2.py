from preprocess import Tokenizer
import xml.etree.ElementTree as ET
from multipledispatch import dispatch
from math import log10
import timeit

class Docs_List:    
    def __init__(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        self.docs_list = []
        for elem in root.iter('DOC'):
            docno = elem.find('DOCNO').text         # type: ignore
            headline = elem.find('HEADLINE').text   # type: ignore
            text = elem.find('TEXT').text           # type: ignore
            self.docs_list.append(Doc(docno, headline, str(headline) + ' ' + str(text)))

class Index:
    def __init__(self, docs_list):
        self.docs_list = docs_list.docs_list
        self.N = len(self.docs_list)
        self.uniq_term_index_dict = {}
        self.no_uniq_term = 0
        self.term_sequence = {}
        self.uniq_term_list = []
        self.docno_list = []
        self.term_ocurring_dict = {}
        self.doc_freq_dict = {}
        self.create_uniq_term_dict()
        self.term_sequence_dict()
        self.write_positional_inverted_index()

    def create_uniq_term_dict(self):
        #creates dictionary mapping a term and the corresponding dim in doc vector
        term_list = []
        self.uniq_term_index_dict = {}
        self.uniq_term_list = []

        for doc in self.docs_list:
            self.docno_list.append(doc.docno)
            for term in doc.terms:
                if not (term in term_list):
                    term_list.append(term)
        self.uniq_term_list = term_list.sort()
        self.no_uniq_term = len(term_list)
        i = 0
        for term in term_list:
            self.uniq_term_index_dict[term] = i
            i += 1
    
    def doc_vector(self, doc_terms):
        doc_vector = [0] * self.no_uniq_term
        for term in doc_terms:
            doc_vector[self.uniq_term_index_dict[term]] += 1
        return doc_vector        
        
    def term_sequence_dict(self):
        term_sequence = {}
        for doc in self.docs_list:
            pos = 0
            for term in doc.terms:      
                if term_sequence.get(term):
                    term_sequence[term].append((doc.docno, pos))
                else:
                    term_sequence[term] = [(doc.docno, pos)]
                pos += 1
        self.term_sequence = term_sequence

    

    def linear_merge(self, posting, posting_other, fun):
        # returns a [(docno, pos)] tuple of docs that satisfy the function 
        # there will be duplicates but we handle these in another method
        arguments = [False, False]
        matches = []
        last_seen = ''
        for tuple in posting:
            docno = tuple[0]
            if docno != last_seen:
                last_seen = docno
                arguments[0] = True
                arguments[1] = self.linear_merge_help(docno, posting_other)
                if fun(arguments[0], arguments[1]):
                    matches.append(tuple)
        return matches
        

    @dispatch(object, object, object)
    def bool_search_new(self, term_x, term_y, fun):  # type: ignore
        ## assume terms are tokenized
        matches = []
        if isinstance(term_x, str):
            tokenizer = Tokenizer()
            tokenizer.load_file_memory(term_x)
            tokenizer.tokenize()
            term_x = tokenizer.terms[0]
            posting_x = self.term_sequence.get(term_x)
        elif isinstance(term_x, list):
            posting_x = term_x
        else:
            raise TypeError

        if isinstance(term_y, str):
            tokenizer = Tokenizer()
            tokenizer.load_file_memory(term_y)
            tokenizer.tokenize()
            term_y = tokenizer.terms[0]
            posting_y = self.term_sequence.get(term_y)
        elif isinstance(term_y, list):
            posting_y = term_y
        else:
            raise TypeError

        if not posting_x:
            posting_x = [(-1, -1)]

        if not posting_y:
            posting_y = [(-1, -1)]

        matches = matches + self.linear_merge(posting_x, posting_y, fun)

        matches = matches + self.linear_merge(posting_y, posting_x, fun)

        return matches

    @dispatch(object, object)
    def bool_search_new(self, term_x, fun):
        matches = []
        if isinstance(term_x, str):
            posting_x = self.term_sequence.get(term_x)
        elif isinstance(term_x, list):
            posting_x = term_x
        else:
            raise TypeError
        if posting_x:
            for docno in self.docno_list:
                term_appears = False
                for tuple in posting_x:
                    if tuple[0] == docno:
                        term_appears = True
                if not term_appears:
                    matches.append((docno, -1))
        else:
            for docno in self.docno_list:
                matches.append((docno, -1))

        return matches


    def linear_merge_help(self, docno, posting):
        for tuple in posting:
            if tuple[0] == docno:
                return True
        return False

    def term_in_doc(self, docno, term):
        posting = self.term_sequence.get(term)
        if posting:
            for tuple in posting:
                if tuple[0] == docno:
                    return True
        return False

    @staticmethod
    def and_wrap(a, b):
        return a and b

    @staticmethod
    def or_wrap(a, b):
        return a or b

    @staticmethod
    def not_wrap(a):
        return not a

    def proximity_search(self, window, term_1, term_2):
        posting_1 = self.term_sequence.get(term_1)
        if not posting_1:
            return []

        posting_2 = self.term_sequence.get(term_2)
        if not posting_2:
            return []

        matches = []
        for tuple_1 in posting_1:
            for tuple_2 in posting_2:
                if tuple_1[0] == tuple_2[0]: # if they occur in the same doc
                    if abs(int(tuple_1[1]) - int(tuple_2[1])) <= int(window):
                        matches.append(tuple_1)

        return matches
        
    def phrase_search(self, phrase):
            posting = self.term_sequence.get(phrase[0])

            if not posting:
                return []

            matches = []
            for tuple in posting:
                match = self.phrase_search_helper(tuple, phrase[1:])
                if match != (-1, -1):
                    matches.append(match)

            return matches


    # ((docno, pos), [(docno, pos)]) -> [(docno, pos)]
    def phrase_search_helper(self, curr_tuple, rest_of_phrase):
        if len(rest_of_phrase) == 0:
            return curr_tuple
        posting = self.term_sequence.get(rest_of_phrase[0])
        if not posting:
            return (-1, -1)
        for tuple in posting:
            if tuple[0] == curr_tuple[0]:
                if tuple[1] == curr_tuple[1] + 1:
                    return self.phrase_search_helper(tuple, rest_of_phrase[1:])
        return (-1, -1)

    def write_positional_inverted_index(self):
        f = open('index.txt', 'w')

        for term in self.term_sequence.keys(): # type: ignore
            df = 0
            matching_docs = []
            hits = []
            posting = self.term_sequence.get(term)  # type: ignore
            for tuple in posting: # type: ignore
                if tuple[0] not in hits:
                    df += 1
                    hits.append(tuple[0])
                    matching_docs.append((tuple[0], [tuple[1]]))
                else: 
                    if matching_docs[len(matching_docs)-1][0] == tuple[0]:
                        matching_docs[len(matching_docs)-1][1].append(tuple[1])
                    else:
                        print('document occurrences out of order - something has gone wrong!')
                        raise Exception

            
            self.term_ocurring_dict.update({term : matching_docs})
            self.doc_freq_dict.update({term : df})
            f.write('{0}:{1}\n'.format(term, df))
            for tuple in matching_docs:
                f.write('\t{0}: '.format(tuple[0]))
                i = 0
                for pos in tuple[1]:
                    f.write(str(pos))
                    if i < len(tuple[1]) -1:
                        f.write(',')
                    else:
                        f.write('\n')
                    i += 1
        f.close()

    def get_term_frequency(self, term, doc):
        doc_pos_tuples = self.term_ocurring_dict.get(term)
        if doc_pos_tuples:
            for tuple in doc_pos_tuples:
                if doc == tuple[0]:
                    return len(tuple[1])
            # casae where term doesn't appear in document
            return 0
        else:
            # case where term doesn't appear in collection
            return 0

    def term_weight(self, term, doc):
        tf = self.get_term_frequency(term, doc)
        if tf == 0:
            return 0
        df = self.doc_freq_dict.get(term)
        if df:
            return (1 + log10(tf)) * log10(self.N / df) 
        else:
            # case where term doesn't appear in collection
            print('something has gone wrong\nterm not in collection')
            raise Exception

    def tfidf_score(self, query, document):
        score = 0
        for term in query:
            score += self.term_weight(term, document)
        return score

    def score_of_all_docs(self, query):
        scores = []
        for doc in self.docno_list:
            score = self.tfidf_score(query, doc)
            if score > 0:
                scores.append((doc, score))
        scores.sort(key=lambda y: y[1], reverse=True)
        return scores[:150]
        


class Doc:
    def __init__(self, docno, headline, text):
        self.docno = docno
        self.headline = headline
        self.text = text
        self.terms = []
        tokenizer = Tokenizer()
        tokenizer.load_file_memory(text)
        self.text = ''
        tokenizer.tokenize()
        self.terms = tokenizer.terms


# this is used to get the index creation time and size for the extra challenge

"""

docs_list = Docs_List('data\\data_2\\trec.5000.xml')

start = timeit.default_timer()
index = Index(docs_list)
stop = timeit.default_timer()

print('Runtime: ', stop - start)

print('Number of Elements in index: ', len(index.term_sequence))

number_of_vals = 0
for group in index.term_sequence.values():
    for tuple in group:
        number_of_vals += 1

print('Number of occurences stored', number_of_vals)

"""
