from model import *

def get_recomendation(corpus):
    documents = [[doc,0] for doc in corpus.document_vectors.values()]
    for doc in documents:
        for a in doc[0].author:
            doc[1] += corpus.authors[a]
        doc[1] /= max(1,len(doc[0].author))
    return sorted(documents,key=lambda item: item[1],reverse=True)[:10]

def update_recomendation(corpus,ranked_documents):
    for doc in ranked_documents:
        for a in doc.document.author:
            corpus.authors[a]+=doc.relevance
    m = max(corpus.authors.values())
    for key in corpus.authors.keys():
        corpus.authors[key] /= m
    save_authors(corpus)

def save_authors(corpus):
    with open('./data/authors.json','w') as file3:
        json.dump(corpus.authors,file3)