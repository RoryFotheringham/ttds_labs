from string import punctuation
from nltk.stem import PorterStemmer
ps = PorterStemmer()
# we want tokenizer to 
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
            #line = line.replace('-', ' ')
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

    def load_and_tokenize_memory(self, data):
        self.load_file_memory(data)
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




    








