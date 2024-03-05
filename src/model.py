from procesing_query import *

class RankedDocument:
    def __init__(self,document,relevance):
        self.document = document
        self.relevance = relevance

def search_documents(final_query,q, corpus):
    ranked_documents = []
    for item in list(corpus.document_vectors.items()):
        ranked_documents.append(RankedDocument(item[1],sim(final_query, corpus,item[1],q)))
    ranked_documents = sorted(ranked_documents,key = lambda item: item.relevance,reverse=True)
    to_return = [ranked_documents[0]]
    
    if ranked_documents[0].relevance == 0:
        return []
    porcent = 85
    
    for i in range(1,len(ranked_documents)):
        if (ranked_documents[i].relevance >= ((porcent*ranked_documents[i-1].relevance)/100)):
            to_return.append(ranked_documents[i])
        else:
            break
    return to_return
    

def sim(final_query,corpus,document,q,simple_version = True,p = 2):
    sum = 0
    for sub_query in final_query:
        sum += 1-math.pow(1-sim_or(sub_query,corpus,document,q,simple_version),p)
    return pow(sum/max(len(final_query),1),1/p)

def sim_or(sub_query,corpus,document,q,simple_version,p = 2):
    if simple_version:
        return sim_or_simple(sub_query,corpus,document,p)
    
    sum = 0
    div = 0
    for word in sub_query:
        if str(word) in document.data.keys():
            sum += math.pow(q[word]*document.data[str(word)],p)
            div += q[word]
    return pow(sum/max(div,1),1/p)

def sim_or_simple(sub_query,corpus,document,p):
    sum = 0
    for word in sub_query:
        if str(word) in document.data.keys():
            sum += math.pow(document.data[str(word)],p)
    return pow(sum/max(len(sub_query),1),1/p)
