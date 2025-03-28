# ADAPTED FROM: https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py
# LICENSE: MIT license
# PAPER: https://arxiv.org/abs/1802.01886

import nltk


def self_bleu_texts(texts: list[str], ngram=3):
    """Rewritten from https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py"""
    tokenized_reference = [nltk.word_tokenize(text) for text in texts]
    weights = tuple((1.0 / ngram for _ in range(ngram)))
    self_bleu_texts = []
    for index, text in enumerate(tokenized_reference):
        hypothesis = text
        other = tokenized_reference[:index] + tokenized_reference[index + 1 :]
        self_bleu = nltk.translate.bleu_score.sentence_bleu(
            references=other,
            hypothesis=hypothesis,
            weights=weights,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
        )
        self_bleu_texts.append(self_bleu)
    return self_bleu_texts
