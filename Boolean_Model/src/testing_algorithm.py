from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import random
from model import *
    
def get_querys(dataset):
    return dataset.queries_iter()

def evaluate_precition(model_classification,right_classification):
    return precision_score(right_classification,model_classification)

def evaluate_recover(model_classification,right_classification):
    return recall_score(right_classification,model_classification)

def evaluate_f1(model_classification,right_classification):
    return f1_score(right_classification,model_classification)

def evaluate_r_precition(model_classification,right_classification,r):
    return precision_score(right_classification, model_classification[:r] + [0] * (len(model_classification) - r))

def evaluate_fallout(model_classification,right_classification):
    matrix = confusion_matrix(right_classification, model_classification)
    print(matrix.ravel())
    #tn, fp, fn, ft = matrix.ravel()
    #return fp / (fp + tn)
    return 0
def evaluate(data_name = "cranfield"):

    dataset = ir_datasets.load(data_name)
    documents = [(doc.doc_id,doc.text) for doc in dataset.docs_iter()]

    tokenized_docs = morphological_reduction_spacy(remove_stopwords_spacy(remove_noise_spacy(tokenization_spacy(documents))), True)

    dictionary = get_dictionary(tokenized_docs)

    tokenized_docs = filter_tokens_by_occurrence(tokenized_docs,dictionary)

    corpus = vector_representation(tokenized_docs, dictionary, [],True)

    querys = get_querys(dataset)

    evaluations_list = [[],[],[],[],[],[],[]]
    for query in querys:
        model_documents = search(query,corpus)
        right_documents_id = get_right_documents(query.query_id,dataset)
        model_classification, right_classification = get_classification(documents,model_documents,right_documents_id)
        evaluations = evaluate_all(model_classification, right_classification)
        print_evaluation(query.text,evaluations)
        for i in range(0,7):
            evaluations_list[i].append(evaluations[i])
    averages = [average(l) for l in evaluations_list]
    print_evaluation("*Total*",averages)
    
def average(l):
    return sum(l)/len(l)

def evaluate_all(model_classification, right_classification):
    p = evaluate_precition(model_classification,right_classification)
    r = evaluate_recover(model_classification,right_classification)
    f1 = evaluate_f1(model_classification,right_classification)
    p1 = evaluate_r_precition(model_classification,right_classification,1)
    p5 = evaluate_r_precition(model_classification,right_classification,5)
    p10 = evaluate_r_precition(model_classification,right_classification,10)
    fo = evaluate_fallout(model_classification,right_classification)
    return [p,r,f1,p1,p5,p10,fo]

def print_evaluation(query,evaluations):
    print(f"""
    Consulta: {query}

    Métricas:
    Precisión: {evaluations[0]}
    Recobrado: {evaluations[1]}
    F1: {evaluations[2]}
    1-Precisión: {evaluations[3]}
    5-Precisión: {evaluations[4]}
    10-Precisión: {evaluations[5]}
    Fallout: {evaluations[6]}
    """)
def get_classification(documents, model_documents, right_documents_id):
    classification = {doc[0]:[0,0] for doc in documents}
    for mdoc in model_documents:
        classification[mdoc][0] = 1
    for rdoc in right_documents_id:
        classification[rdoc][1] = 1
    model_classification = [classification[key][0] for key in classification]
    right_classification = [classification[key][1] for key in classification]
    return  model_classification, right_classification


def get_right_documents(query_id,dataset):
    return (
    [
      doc_id
      for (queryt_id, doc_id, relevance, _) in dataset.qrels_iter()
      if queryt_id == query_id and relevance in [3, 4]
    ])

evaluate()