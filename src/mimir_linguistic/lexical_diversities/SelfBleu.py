# ADAPTED FROM: https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py
# LICENSE: MIT license
# PAPER: https://arxiv.org/abs/1802.01886

import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from abc import abstractmethod
nltk.download("punkt")
class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

class SelfBleu(Metrics):
    def __init__(self, test_text=[], gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            for text in self.test_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

    def get_bleu_one_hypothesis(self, index=0):
        ngram = self.gram
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        hypothesis = reference[index]
        other = reference[:index] + reference[index+1:]
        bleu = self.calc_bleu(other, hypothesis, weight)
        return bleu