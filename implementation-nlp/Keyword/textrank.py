from src import utils
import numpy as np
import math
from collections import Counter
from pyvi import ViPosTagger
import string


def calculated_weighted_edges(preprocess_text, window_size):
    vocabulary = list(set(preprocess_text))
    N = len(vocabulary)
    print('Lenght of vocabulary: %d' % N)
    weighted_edges = np.zeros((N, N), dtype=np.float32)
    covered = []
    for i in range(0, N):
        for j in range(0, N):
            if j == i:
                weighted_edges[i][j] = 0
            else:
                for start in range(0, N - window_size):
                    end = start + window_size
                    window = preprocess_text[start:end]
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        index_i = start + window.index(vocabulary[i])
                        index_j = start + window.index(vocabulary[j])
                        if [index_i, index_j] not in covered:
                            weighted_edges[i][j] += 1/math.fabs(index_i - index_j)
                            covered.append([index_i, index_j])
    return weighted_edges

def calculated_weighted_edges_ver2(preprocess_text, window_size):
    word_counts = Counter(preprocess_text)
    TF = {}
    for key in word_counts:
        TF[key] = word_counts[key]/len(preprocess_text)
    vocabulary = list(set(preprocess_text))
    N = len(vocabulary)
    print('Lenght of vocabulary: %d' % N)
    weighted_edges = np.zeros((N, N), dtype=np.float32)
    covered = []
    for i in range(0, N):
        for j in range(0, N):
            if j == i:
                weighted_edges[i][j] = 0
            else:
                for start in range(0, N - window_size):
                    end = start + window_size
                    window = preprocess_text[start:end]
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        frequency_i = TF[vocabulary[i]]
                        frequency_j = TF[vocabulary[j]]
                        if [vocabulary[i], vocabulary[j]] not in covered:
                            weighted_edges[i][j] = frequency_i*frequency_j/(frequency_j + frequency_j)
                            covered.append([vocabulary[i], vocabulary[j]])
    return weighted_edges


def calculated_inout(weighted_edges):
    N = weighted_edges.shape[0]
    inout = np.zeros((N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            inout += weighted_edges[i][j]
    return inout

def calculated_weighted_vertices(inout, weighted_edges, threshold=0.000001):
    MAX_LOOP = 1000
    d = 0.85
    N = weighted_edges.shape[0]
    scores = np.ones((N), dtype=np.float32)
    for iter in range(MAX_LOOP):
        prev_scores = np.copy(scores)
        for i in range(0, N):
            summation = 0
            for j in range(0, N):
                if weighted_edges[i][j] != 0.0:
                    summation += (weighted_edges[i][j]/inout[j])*scores[j]
            scores[i] = d + (1 - d)*summation
        if np.sum(np.fabs(prev_scores - scores)) <= threshold:
            print('Ends at step: ' + str(iter) + ' ...')
            break
    return scores

def get_keys(scores, vocabulary, size=10):
    index = np.flip(np.argsort(scores), 0)
    key_words = []
    for i in range(size):
        key_words.append(vocabulary[int(index[i])])
    return key_words

def gen_candidate(text, vocabulary, scores):
    stop_words = load_new_stopwords(text)
    phrase = ''
    phrases = []
    text = utils.preprocess(text, stop_word=[])
    for word in text:
        if word in stop_words:
            if phrase != '' and phrase not in phrases:
                phrases.append(phrase)
                phrase = ''
        else:
            phrase += word + ' '
    scores_phrases = {}
    for phrase in phrases:
        score = 0.0
        for word in phrase.split():
            score += scores[vocabulary.index(word)]
        scores_phrases[phrase] = score
    return scores_phrases

def load_new_stopwords(text):
    not_labels = ['A', 'L', 'R', 'T', 'E', 'M', 'I']
    stop_words = utils.load_stop_words('stopwords.txt')
    stop_words += list(string.punctuation)
    stop_word_mini = []
    text = text.split()
    tokens, postag = ViPosTagger.postagging_tokens(text)
    for i in range(len(tokens)):
        if postag[i] in not_labels:
            stop_word_mini.append(tokens[i].lower())
    stop_words += stop_word_mini
    stop_words = set(stop_words)
    return stop_words

def get_keywords(text, size):
    preprocess_text = utils.preprocess(text, stop_word=utils.load_stop_words('stopwords.txt'))
    print('Lenght of text after preprocessing: %d' % len(preprocess_text))
    vocabulary = list(set(preprocess_text))
    weighted_edges = calculated_weighted_edges_ver2(preprocess_text, 8)
    inout = calculated_inout(weighted_edges)
    scores = calculated_weighted_vertices(inout, weighted_edges, threshold=0.0)
    result = get_keys(scores, vocabulary, size=size)
    for key in result:
        print(key, end='\t-\t')
    print('\n')

from src import tf
from pyvi import ViTokenizer

if __name__ == '__main__':
    corpus = utils.parse_file('12.txt')
    f = open('b.txt', 'r')
    sentences = []
    for line in f:
        if line.strip():
            sentences.append(line.strip())
    text = ''
    for sent in sentences:
        text += str(sent) + ' '
    print(text)
    text = ViTokenizer.tokenize(text)
    print('--------------------------------------')
    get_keywords(text, 10)
    print('--------------------------------------')
    tf.keyword_extraction(text, 10)



