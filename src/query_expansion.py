from pre_procesamiento import *
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import Levenshtein
import re
import math


def expand_query(doc_query, corpus):
    near_words = get_near_words(doc_query,corpus.dictionary,corpus.coo_matrix)
    
    doc_query.add_words(near_words)
    
    expanded_query = get_words_from_synsets(get_hypernyms(get_synsets(doc_query)))
    
    expanded_query = [filter_by_occurrence(l,["" for w in l],corpus.vocabulary())[0] for l in expanded_query]
    q_near_words = [(id,freq/10) for id,freq in corpus.dictionary.doc2bow(near_words)]
    return expanded_query, q_near_words

def get_near_words(query,dictionary,matrix, n = 1):
    words_scores = []
    for word in dictionary.token2id:
        score = 0
        for qword in query.data:
            key = tuple(sorted([dictionary.token2id[word],dictionary.token2id[qword]]))
            if key in matrix.keys():
                score += matrix[key]
        words_scores.append((word,score))
    
    return [w[0] for w in sorted(words_scores,key=lambda word_score: word_score[1],reverse=True)][:n]

pos_tag_map = {
    'NOUN': [ wn.NOUN ],
    'ADJ': [ wn.ADJ, wn.ADJ_SAT ],
    'ADV': [ wn.ADV ],
    'VERB': [ wn.VERB ]
}

def get_synsets(doc_query):
    synsets = []
    for i in range(0,len(doc_query.data)):
        synsets.append([doc_query.data[i]])
        if doc_query.tags[i] not in pos_tag_map.keys():
            continue
        else:
            synsets[i].extend(wn.synsets(doc_query.data[i], pos_tag_map[doc_query.tags[i]]))
    return synsets

def get_hypernyms(synsets):
    for synset in synsets:
        hypers = []
        for i in range(1,len(synset)):
            hypers.extend(synset[i].hypernyms())
        synset.extend(hypers)
    return synsets

def get_words_from_synsets(synsets):
    tokens = []
    for synset in synsets:
        tokens.append([synset[0]])
        for i in range(1,len(synset)):
            w = synset[i].name().split('.')[0]
            if w in tokens[len(tokens)-1]:
                continue
            else:
                tokens[len(tokens)-1].append(w)
    return [underscore_replacer(l) for l in tokens]

def get_most_similar(word,vocanulary):
    current_distance = 10000
    current_word = ""
    for w in vocanulary:
        new_distance = Levenshtein.distance(word,w)
        if(new_distance < current_distance):
            current_distance = new_distance
            current_word = w
    return current_word

def underscore_replacer(tokens):
    new_tokens = []
    for token in tokens:
        new_token = re.sub(r'_', ' ', token)
        new_tokens.append(new_token)
    return new_tokens

def print_not_founded_info(not_founded,corpus):
    if len(not_founded) == 0:
        return []
    to_replace = [get_most_similar(word,corpus.vocabulary()) for word in not_founded]
    print("No se encontraron las palabras:")
    for w in not_founded:
        print(w)
    print("Estás buscando?:")
    for w in to_replace:
        print(w)
    return to_replace