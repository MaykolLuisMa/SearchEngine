import ir_datasets
import nltk
import spacy
import gensim
from sympy import sympify, to_dnf, Not, And, Or

nlp = spacy.load("en_core_web_sm")
def load_dataset():
    dataset = ir_datasets.load("cranfield")
    return [doc.text for doc in dataset.docs_iter()]

def tokenization_spacy(texts):
  return [[token for token in nlp(doc)] for doc in texts]

def remove_noise_spacy(tokenized_docs):
  return [[token for token in doc if token.is_alpha] for doc in tokenized_docs]

def remove_stopwords_spacy(tokenized_docs):
  stopwords = spacy.lang.en.stop_words.STOP_WORDS
  return [
      [token for token in doc if token.text not in stopwords] for doc in tokenized_docs
  ]

def morphological_reduction_spacy(tokenized_docs, use_lemmatization=True):
  stemmer = nltk.stem.PorterStemmer()
  return [
    [token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc]
    for doc in tokenized_docs
  ]
def filter_tokens_by_occurrence(tokenized_docs, no_below=5, no_above=0.5):
  global dictionary
  dictionary = gensim.corpora.Dictionary(tokenized_docs)
  dictionary.filter_extremes(no_below=no_below, no_above=no_above)

  filtered_words = [word for _, word in dictionary.iteritems()]
  filtered_tokens = [
      [word for word in doc if word in filtered_words]
      for doc in tokenized_docs
  ]

  return filtered_tokens

def build_vocabulary(dictionary):
  vocabulary = list(dictionary.token2id.keys())
  return vocabulary

def vector_representation(tokenized_docs, dictionary, vector_repr, use_bow=True):
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    if use_bow:
        vector_repr = corpus
    else:
        tfidf = gensim.models.TfidfModel(corpus)
        vector_repr = [tfidf[doc] for doc in corpus]

    return vector_repr