# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:41:34 2023

@author: Mai Al-Khatib
"""

# in the main function that calls model_static, generate two version of beta, siml, and simh
# one version is generated using the word2vec and another which uses speech2vec
# call model_static function sending it the beta, siml, and simh generated with word2vec embeddings along with the global frequency data
# save the function call in variable logword2vec
# call model_static function sending it the beta, siml, and simh generated with speech2vec embeddings along with the global frequency data
# save the function call outcome in variable logspeech2vec
# if logword2vec > logspeech2vec then choose logword2vec as a one with better fit
# else if logspeech2vec > logword2vec then choose logspeech2vec as one with better fit
# else if logspeech2vec == logword2vec then both models are equally preferable 