import numpy as np
import pickle
from pyvi import ViTokenizer
import re


correct_mapping = {
    "ship": "vận chuyển",
    "shop": "cửa hàng",
    "m": " mình ",
    "mik": " mình ",
    "k": " không ",
    "kh": " không ",
    "tl": " trả lời ",
    "r": " rồi ",
    "fb": " mạng xã hội ",
    "face": " mạng xã hội ",
    "thanks": " cảm ơn ",
    "thank": "cảm ơn",
    "tks": " cảm ơn ",
    "tk": " cảm ơn ",
    "j": "gì"
}

def remove_special(text):
    sent = re.sub("%|:|'|,|\"|\(|\) |\)|\*|-|(http\S+)|(@\S+)|RT|\#|!|:|\.|[0-9]|\/|\. |\.|\“|’s|;|–|” |\\n|&|-|--",'', text)
    sent = sent.split()
    for i in range(len(sent)):
        if sent[i] in correct_mapping:
            sent[i] = correct_mapping[sent[i]]
    return ' '.join(sent)

def preprocess(sentence):
    sentence = remove_special(sentence)
    stop_words = load_stop_words()
    new_sentences = []
    sentence = ViTokenizer.tokenize(sentence)
    sentence = sentence.split()
    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            new_sentences.append(word)
    return new_sentences


def load_stop_words():
    f = open('stopwords.txt', 'r')
    stop_word = []
    for line in f:
        stop_word.append(line.strip())
    return stop_word


def parse_file(file_name):
    sentences = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            labels.append(line[0])
            sentences.append(preprocess(line[-1]))
    return sentences, labels


def build_dict(sentences):
    DICT = {}
    count = 0
    print('Building dictionary !')
    for sent in sentences:
        for word in sent:
            if word not in DICT:
                DICT[word] = count
                count += 1
            else:
                continue
    pickle.dump(DICT, open('DICT', 'wb'))
    print('Done !')

def load_dict():
    try:
        DICT = pickle.load(open('DICT', 'rb'))
        return DICT
    except:
        return {}

def bag_of_word(sentence, DICT):
    vector = np.zeros(len(DICT))
    for token in sentence:
        if token in DICT:
            vector[DICT[token]] += 1
        else:
            continue
    return vector

if __name__ == '__main__':
    parse_file('train.txt')
