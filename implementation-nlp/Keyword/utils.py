import re
import pickle
import collections
import math

def load_stop_words(file_name):
    stop_word = []
    with open(file_name, 'r') as f:
        for line in f:
            stop_word.append(line.strip())
    return stop_word


def parse_file(file_name):
    corpus = []
    with open(file_name, 'r') as f:
        for line in f:
            corpus.append(line.strip())
    return corpus


def preprocess(text, stop_word):
    sent = re.sub(" %|:|'|,|\"|\(|\) |\)|\*|-|(http\S+)|(@\S+)|RT|\#|!|:|\.|[0-9]|\/|\. |\.|\“|’s|;|–|” |\\n|&|-|--[^\w\s]|[?]|‘",'', text)
    new_sent = []
    for word in sent.split():
        if word.lower() not in stop_word:
            new_sent.append(word.lower())
    return new_sent


def build_dict():
    stop_word = load_stop_words('stopwords.txt')
    DICT = {}
    print('Building dictionary !')
    corpus = parse_file('12.txt')
    for line in corpus:
        line = preprocess(line, stop_word)
        for word in line:
            if word not in DICT:
                DICT[word] = 0
    pickle.dump(DICT, open('DICT', 'wb'))
    print('Done !')


def load_dict(file_name):
    try:
        DICT = pickle.load(open(file_name, 'rb'))
        return DICT
    except:
        return {}

def IDF(file_name):
    stop_word = load_stop_words('stopwords.txt')
    IDF = load_dict('DICT')
    total = 0
    corpus = parse_file(file_name)
    for line in corpus:
        line = preprocess(line, stop_word)
        line = dict(collections.Counter(line))
        total += 1
        for key in line:
            if key in IDF:
                IDF[key] += 1
            else:
                continue
    for key in IDF:
        IDF[key] = math.log10(total/(1 + IDF[key]))
    pickle.dump(IDF, open('IDF', 'wb'))


if __name__ == '__main__':
    build_dict()
    IDF('12.txt')




