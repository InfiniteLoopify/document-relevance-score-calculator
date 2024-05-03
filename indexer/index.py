import numpy as np
import os
import re
from nltk.stem import WordNetLemmatizer


class Indexer:
    def __init__(self):
        self.vsm = None
        self.features = {}
        self.idf = None
        self.stop_words = []
        self.vsm_init_size = 7000
        self.lemmatize = WordNetLemmatizer()
        self.lemmatize.lemmatize('')

    def read_stop_words(self, path):
        """ read stopwords from file """

        self.stop_words = []
        with open(path, 'r') as f:
            for line in f:
                for word in line.split():
                    self.stop_words.append(word)

    def read_file(self, speech_path, file_path):
        """ read array files, if they exist, else read from speeches """

        # check if vsm.npy features.txt idf.npy exist
        if os.path.isfile(file_path+'vsm.npy') and\
           os.path.isfile(file_path+'idf.npy') and\
           os.path.isfile(file_path+'features.npy'):
            self.vsm = np.load(file_path+'vsm.npy')
            self.idf = np.load(file_path+'idf.npy')
            self.features = np.load(
                file_path+'features.npy', allow_pickle='TRUE').item()
        else:
            self.read_from_speeches(speech_path, file_path)

    def read_from_speeches(self, speech_path, file_path):
        """ read all speech documents one by one and tokenize words """

        # read and save stop words from file
        self.read_stop_words(file_path+'stopword_list.txt')

        # generate file list from directory provided
        file_list = os.listdir(speech_path)
        file_list = sorted(file_list, key=lambda x: int(
            "".join([i for i in x if i.isdigit()])))

        # initialize vector space model with zeros
        self.vsm = np.zeros(shape=(len(file_list), self.vsm_init_size))

        # open every document, and index the words in them
        tokenize_regex = r"[^\w]"
        for doc_id, file_name in enumerate(file_list):
            with open(speech_path + file_name, 'r') as file_data:

                # ignore first line of every file
                file_data.readline()
                for line in file_data:

                    # split and tokenize word by given charecters
                    for word in re.split(tokenize_regex, line):
                        self.tokenize(word, doc_id)

        # delete unused columns in the vector
        self.delete_extra_cols()

        # calculate document frequency and idf using df
        df = np.count_nonzero(self.vsm > 0, axis=0)
        N = len(self.vsm)
        self.idf = (np.log10(df / N)) * -1

        # write vsm, features and idf to file
        np.save(file_path+'vsm.npy', self.vsm)
        np.save(file_path+'idf.npy', self.idf)
        np.save(file_path+'features.npy', self.features)

    def tokenize(self, word, doc_id):
        """ tokenize and insert term frequency in vsm """

        # remove trailing commas and apostrophes and lower word
        word = re.sub('[,\'\n]', '', word)
        word = word.lower()

        # apply lemmatization algo to each word
        word = self.lemmatize.lemmatize(word)

        # if word is not stopword, increment vsm value
        if word and word not in self.stop_words:
            if word not in self.features:
                self.features[word] = len(self.features)

            # resize vsm if feature count exceed its' current size
            if len(self.features) > len(self.vsm[0]):
                self.insert_extra_cols()
            self.vsm[doc_id][self.features[word]] += 1

    def delete_extra_cols(self):
        """ delete empty and unused columns from vsm """

        start = len(self.features)
        stop = len(self.vsm[0])
        self.vsm = np.delete(
            self.vsm, [i for i in range(start, stop)], axis=1)

    def insert_extra_cols(self):
        """ resize vsm if features start to overflow """

        rows = len(self.vsm)
        cols = self.vsm_init_size
        self.vsm = np.concatenate(
            (self.vsm, np.zeros(shape=(rows, cols))), axis=1)

    def calculate(self, query, alpha=0.0005):
        """ calculate tf-idf for given query using provided alpha value """

        # split string and lemmatize words
        query = [self.lemmatize.lemmatize(word.lower())
                 for word in query.split()]

        # for every word in string, update the query list
        query_list = np.zeros(len(self.vsm[0]))
        for word in query:
            if self.features.get(word) != None:
                query_list[self.features[word]] += 1

        relevancy_list = []

        # if query list is empty
        if not np.any(query_list):
            return []

        # update query list with tf * idf
        query_list = np.multiply(query_list, self.idf)

        # for every document, calculate cosine similarity with query
        for doc in self.vsm:
            doc_tfidf = np.multiply(doc, self.idf)
            dot_product = np.dot(doc_tfidf, query_list)
            doc_magnitude = np.linalg.norm(doc_tfidf)
            query_magnitude = np.linalg.norm(query_list)
            relevancy_list.append(
                dot_product/(doc_magnitude*query_magnitude))

        # sort the final relevancy list containing values of cosine similarity and document id
        relevancy_list = sorted([(i, x)
                                 for i, x in enumerate(relevancy_list) if x >= alpha], key=lambda x: x[1], reverse=True)
        return relevancy_list


if __name__ == "__main__":
    indexer = Indexer()
    indexer.read_file('speech/', 'files/')

    # x = indexer.calculate('pakistan afghanistan')
    # x = indexer.calculate('personnel policies')
    # x = indexer.calculate('develop solutions')
    x = indexer.calculate('developments praised')
    # x = indexer.calculate('muslims')
    # x = indexer.calculate('no patience for injustice')
    # x = indexer.calculate('pakistan afghanistan aid')
    # x = indexer.calculate('biggest plane wanted hour')

    for elem in x:
        print("%d:\t%0.4f" % (elem[0], elem[1]))
    # print(len(indexer.vsm[0]))
