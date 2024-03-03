import ir_datasets
import nltk
import spacy
import gensim
from sympy import sympify, to_dnf, Not, And, Or, symbols

nlp = spacy.load("en_core_web_sm")
def load_dataset(data_name):
    dataset = ir_datasets.load(data_name)
    return [(doc.doc_id,doc.text) for doc in dataset.docs_iter()]

def tokenization_spacy(texts):
  return [(id,[token for token in nlp(doc)]) for (id,doc) in texts]

def remove_noise_spacy(tokenized_docs):
  return [(id,[token for token in doc if token.is_alpha]) for id,doc in tokenized_docs]

def remove_stopwords_spacy(tokenized_docs):
  stopwords = spacy.lang.en.stop_words.STOP_WORDS
  return [
      (id,[token for token in doc if token.text not in stopwords]) for id,doc in tokenized_docs
  ]

def morphological_reduction_spacy(tokenized_docs, use_lemmatization=True):
  stemmer = nltk.stem.PorterStemmer()
  return [
    (id,[token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc])
    for id,doc in tokenized_docs
  ]

def get_dictionary(tokenized_docs,no_below=5, no_above=0.5):
  dictionary = gensim.corpora.Dictionary([doc for _,doc in tokenized_docs])
  dictionary.filter_extremes(no_below=no_below, no_above=no_above)
  return dictionary

def filter_tokens_by_occurrence(tokenized_docs,dictionary):
  
  filtered_words = [word for _, word in dictionary.iteritems()]
  filtered_tokens = [
      (id,[word for word in doc if word in filtered_words])
      for id,doc in tokenized_docs
  ]

  return filtered_tokens

def build_vocabulary(dictionary):
  vocabulary = list(dictionary.token2id.keys())
  return vocabulary

def vector_representation(tokenized_docs, dictionary, vector_repr, use_bow=True):
    corpus = [(id,dictionary.doc2bow(doc)) for id,doc in tokenized_docs]

    if use_bow:
        vector_repr = corpus
    else:
        tfidf = gensim.models.TfidfModel(corpus)
        vector_repr = [tfidf[doc] for doc in corpus]

    return vector_repr