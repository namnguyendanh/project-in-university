from keras.models import model_from_json
from character import CharacterModel
from process import gen_ngram_from_text
from preprocess import remove_accent
import numpy as np
from collections import Counter

import config
import pickle
import re
import string
import logging
import sys

logging.getLogger("tensorflow").setLevel(logging.WARNING)


class ToneModel(object):
    def __init__(self, config_file, model_config, weights_file, alphabet_file):
        self.maxlen = config_file.MAXLEN
        self.ngram = config_file.NGRAM
        try:
            json_file = open(model_config, "r")
            model_json = json_file.read()
            self.models = model_from_json(model_json)
            self.models.load_weights(weights_file)
        except Exception as e:
            assert e
            sys.exit(1)
        try:
            index_alphabet = pickle.load(open(alphabet_file, "rb"))
        except Exception as e:
            assert e
            sys.exit(1)
        self.codec = CharacterModel(index_alphabet=index_alphabet, maxlen=self.maxlen)

    def add_tone_ngram(self, ngram):
        text = ' '.join(ngram)
        n = len(text)
        if n < self.maxlen:
            text += "\x00" * (self.maxlen - n)
        pred = self.models.predict(np.array([self.codec.encode(remove_accent(text))]), verbose=0)
        return self.codec.decode(pred[0]).strip('\x00')

    def add_tone_phrase(self, phrase):
        ngrams = list(gen_ngram_from_text(phrase.lower(), n=self.ngram))
        guessed_ngram = list(self.add_tone_ngram(ngram) for ngram in ngrams)
        candidates = [Counter() for _ in range(len(phrase.split()))]
        for nid, ngram in enumerate(guessed_ngram):
            for wid, word in enumerate(re.split(' +', ngram)):
                candidates[nid + wid].update([word])
        try:
            output = " ".join(c.most_common(1)[0][0] for c in candidates)
        except Exception as e:
            output = ""
            assert e
        return output

    def add_tone(self, sentence):
        y_true = 0
        y_pred = 0
        index = 0
        output = ""
        sentence = sentence.strip()
        m = len(sentence)
        for i, c in enumerate(sentence):
            f = False
            if c in string.punctuation or i == (m - 1):
                if i == (m - 1) and c not in string.punctuation:
                    i = m
                    f = True
                phrase = sentence[index: i]
                n = len(phrase.split())
                if n < 2:
                    output += phrase + c
                    y_true += n + 1
                    y_pred += n + 1
                else:
                    out = self.add_tone_phrase(phrase.lower()).strip()
                    flag = False
                    if phrase[0] == " ":
                        output += " "
                    if phrase[-1] == " ":
                        flag = True
                    phrase = phrase.strip().split()
                    out = out.strip().split()
                    try:
                        for token_1, token_2 in zip(phrase, out):
                            for j, k in enumerate(token_1):
                                if k.isupper():
                                    output += token_2[j].upper()
                                else:
                                    output += token_2[j]
                            if token_1.lower() == token_2.lower():
                                y_pred += 1
                            y_true += 1
                            output += " "
                        output = output.strip()
                    except Exception as e:
                        print("\tInput: " + phrase)
                        print("\tOutput: " + out)
                        print("=" * 50)
                        assert e
                    if flag:
                        output += " "
                    if not f:
                        output += c
                    y_pred += 1
                    y_true += 1
                index = i + 1
        return y_pred, y_true, output


text = """Th?? sinh ch??? ???????c ??i???u ch???nh ????ng k?? x??t tuy???n m???t l???n v?? ch??? ???????c s??? d???ng m???t trong hai ph????ng th???c tr???c \
tuy???n ho???c b???ng phi???u. V???i ??i???u ch???nh b???ng ph????ng th???c tr???c tuy???n, c??c em s??? d???ng t??i kho???n v?? m???t kh???u c?? nh??n ???? \
???????c c???p. Ph????ng th???c n??y ch??? ch???p nh???n khi s??? l?????ng nguy???n v???ng sau khi ??i???u ch???nh kh??ng l???n h??n s??? ???? ????ng k?? ban \
?????u. """
text2 = """Ninh D????ng Lan Ng???c sinh ng??y 4/4/1990 ??? TP HCM. N??m 2010, c?? ???????c bi???t ?????n l???n ?????u qua b??? phim C??nh ?????ng\
 b???t t???n. Nh??? g????ng m???t s??ng, di???n xu???t t??? nhi??n, c?? ???????c v?? l?? "ng???c n???" khi b?????c v??o l??ng gi???i tr?? Vi???t."""
text3 = """B??? tr?????ng Qu???c ph??ng Chetta Thanajaro kh???ng ?????nh"""
if __name__ == "__main__":
    model_file = "models/modelv2/model_v2.json"
    weights_file = "models/modelv2/best_model_v2.hdf5"
    alphabet_file = "idxabc.pickle"
    model = ToneModel(config, model_file, weights_file, alphabet_file)
    print(model.add_tone(text2))
