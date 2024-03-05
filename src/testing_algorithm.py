from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from recomendation_system import *
import random
    
def evaluate(data_name = "cranfield"):
    
    corpus = Corpus()

    dataset = ir_datasets.load(data_name)

    querys = get_querys(dataset)

    evaluations_list = [[],[],[],[]]
    for i,query in enumerate(querys):

        print("-------------------------------------------------------")
        print("the query: ")
        print(query.text)
        final_query,q = process_query(query.text,corpus)        
        model_documents = search_documents(final_query,q,corpus)

        right_documents_id = get_right_documents(query.query_id,dataset)
        if(len(right_documents_id) == 0):
            continue
        model_classification, right_classification = get_classification(corpus.document_vectors,model_documents,right_documents_id)
        evaluations = get_stats(get_confusion_matrix(right_classification,model_classification))
        print_evaluation(query.text,evaluations)
        for i in range(0,len(evaluations_list)):
            evaluations_list[i].append(evaluations[i])
        

    averages = [average(l) for l in evaluations_list]
    print_evaluation("*Total*",averages)
    
def average(l):
    return sum(l)/len(l)


def print_evaluation(query,evaluations):
    print(f"""
    Consulta: {query}

    Métricas:
    Precisión: {evaluations[0]}
    Recobrado: {evaluations[1]}
    F1: {evaluations[2]}
    Fallout: {evaluations[3]}
    """)
def get_classification(documents, model_documents, right_documents_id):
    print("lenths")
    print(len(documents))
    print(len(right_documents_id))
    print(len(model_documents))
    classification = {doc:[0,0] for doc in documents.keys()}
    for mdoc in model_documents:
        classification[mdoc.document.id][0] = 1
    for rdoc in right_documents_id:
        classification[rdoc][1] = 1
    model_classification = [classification[key][0] for key in classification.keys()]
    right_classification = [classification[key][1] for key in classification.keys()]
    return  model_classification, right_classification


def get_right_documents(query_id,dataset):
    return (
    [
      doc_id
      for (queryt_id, doc_id, relevance, _) in dataset.qrels_iter()
      if queryt_id == query_id and relevance in [1,2,3, 4]
    ])

def get_querys(dataset):
    return dataset.queries_iter()

def get_limited_dataset(data):
    documents = [Document(doc.id,doc.title,doc.text) for doc in data.docs_iter()]
    random.shuffle(documents)
    recovered_documents = documents[:random.randint(1, len(documents) - 1)]
    return recovered_documents

def get_confusion_matrix(right_classification,model_classification):
    matrix = confusion_matrix(right_classification,model_classification).ravel()
    return tuple(matrix)

def get_stats(matrix):
    tn, fp, fn, tp = matrix
    precition = tp/(tp+fp)
    recover = tp/(tp+fn)
    if tp + fn == 0:
        recover = 0
    f1 = (2*precition*recover)/(precition+recover)
    if (precition+recover) == 0:
        f1 = 0
    fallout = fp/(fp+tn)
    return [precition,recover,f1,fallout]

evaluate()