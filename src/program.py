from recomendation_system import *

def ejecution():
    print("Ejecution start")
    #build_dataset("cranfield")
    corpus = Corpus()
    while True:
        recomendation = get_recomendation(corpus)
        #for doc in recomendation:
        #    print(doc[0].id,doc[1])
        print("Introduce your query")
        query = input()
        final_query,q = process_query(query,corpus)
        result = search_documents(final_query,q,corpus)
        for doc in result:
            print(doc.document.id,doc.document.title,doc.relevance)
        update_recomendation(corpus,result)

def test():
    dataset = ir_datasets.load("cranfield")
    dataset = dataset.docs_iter()
    l = []
    for i,data in enumerate(dataset):
        print(data.author)
        l.append(data.author)
        #print(data.bib)
        if i == 10:
            break
    print("----")
    l = [nlp(t) for t in l]
    for tokens in l:
        print(authors_parser([token.text for token in tokens]))
def rest2():
    dataset = ir_datasets.load("cranfield")
    dataset = dataset.docs_iter()
    corpus = Corpus()
    no_author = []
    for doc in corpus.document_vectors.values():
        if len(doc.author)==0:
            no_author.append(doc.id)
    for doc in dataset:
        if doc.doc_id in no_author:
            print(doc.author)
def test3():
    query = [[(1,5),(2,10),(3,15),(4,0.5)],[]]
    tf_idf = gensim.models.TfidfModel(query,normalize = True)
    print(tf_idf[query[0]])
    print(tf_idf.idfs)

    print("------------")
    query = [[(1,5),(2,10),(3,15),(4,0.5)],[]]
    tf_idf = gensim.models.TfidfModel(query,normalize = False)
    print(tf_idf[query[0]])
    print(tf_idf.idfs)

ejecution()
def test5():
    def get_right_documents(query_id,dataset):
        return [
        doc_id
        for (queryt_id, doc_id, relevance, _) in dataset.qrels_iter()
        if queryt_id == query_id and relevance in [1,2,3, 4]
        ]

    data = ir_datasets.load("cranfield")
    querys = data.queries_iter()
    an = []
    for id,text in querys:
        an.append((id,get_right_documents(id,data)))
    print(len(an))
    print(len([i for i,d in an if len(d)==0]))