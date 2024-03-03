from procesing_query import *

def ejecution():
    print("Ejecution start")
    build_dataset("cranfield")
    corpus = Corpus()
    while True:
        print("Introduce your query")
        query = input()
        final_query = process_query(query,corpus)
        print(final_query)
        result = search_documents(final_query,corpus)
        for doc in result:
            print(doc.document.title,doc.document.id,doc.relevance)
dataset = ir_datasets.load("cranfield")
dataset = dataset.docs_iter()
for data in dataset:
    #print(data.author)
    print(data.bib)