from src import utils
from collections import Counter
import math
from underthesea import ner, pos_tag

def tf(text):
    text = utils.preprocess(text, stop_word=utils.load_stop_words('stopwords.txt'))
    N = len(text)
    text = dict(Counter(text))
    TF = {}
    for key in text:
        if key not in TF:
            TF[key] = text[key]/N
    return TF
def keyword_extraction(text, size=10):
    IDF = utils.load_dict('IDF')
    TF = tf(text)
    for key in TF:
        if key not in IDF:
            TF[key] = TF[key]*math.log10(2000/1)
        else:
            TF[key] = TF[key]*IDF[key]
    keys = sorted(TF.keys(), key=lambda x: TF[x], reverse=True)
    count = 0
    for key in keys:
        if count == size:
            break
        else:
            print(key, end='\t-\t')
            count += 1

if __name__ == '__main__':
    sentences = utils.parse_file('12.txt')
    print(sentences[10])
    print(pos_tag(sentences[10]))
