from lexer import *

def query_to_dnf(query):
    # TODO: your code goes here!
    processed_query = ""
    
    # Convertir a expresión sympy y aplicar to_dnf
    query_expr = sympify(processed_query, evaluate=False)
    query_dnf = to_dnf(query_expr, simplify=True)

    return query_dnf

def get_matching_docs(query_dnf,corpus,dictionary):

    # Función para verificar si un documento satisface una componente conjuntiva de la consulta
    matching_documents = []

    return matching_documents