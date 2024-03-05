from query_expansion import *
def process_query(query,corpus, search_alternative = False, expand = False):
    doc_query = Document(None,'query',query)
    doc_query.process()
    
    not_founded = doc_query.filter_by_occurrence(corpus.vocabulary())
    to_replace = print_not_founded_info(not_founded,corpus)
    
    if len(doc_query.data) == 0:
        if search_alternative:
            doc_query.add_words(to_replace)
        else:
            return []
    q = corpus.dictionary.doc2bow(doc_query.data)
    doc_query.data = [corpus.dictionary[id] for id,_ in q] 
    
    expanded_query = []
    q_near_words = []
    if expand:
        expanded_query,q_near_words = expand_query(doc_query,corpus)
    else:
        expanded_query = [[word] for word in doc_query.data]
    
    print("expanded_query:")
    print(expanded_query)
    
    expanded_query = [[w[0] for w in corpus.dictionary.doc2bow(l)] for l in expanded_query]
    
    q = calculate_tfidf_query(q,q_near_words,expanded_query,corpus)

    return expanded_query,q

def calculate_tfidf_query(q,q_near_words,expanded_query,corpus):
    q_near_words = [(id,seq) for id,seq in q_near_words if id not in [item for item,_ in q]]
    q.extend(q_near_words)
    for items in expanded_query:
        for item in items:
            if item not in [w for w,_ in q]:
                q.append((item,0.5))
    q = gensim.models.TfidfModel([q,[]],normalize = True)[q]
    q = {id: tf*corpus.word_vectors[id].idf for id,tf in q}
    return q

