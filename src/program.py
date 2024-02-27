from procesing_query import *

build_dataset("cranfield")
corpus = get_corpus()
while True:
    query = input()
    final_query = process_query(query,corpus)
    result = search_documents(final_query,corpus)
    for doc in result:
        print(doc.title)