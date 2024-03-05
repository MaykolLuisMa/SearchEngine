import streamlit as st
from recomendation_system import *
import pandas as pd

import os
def f():
    st.title("Tansakugami II")
    corpus = Corpus()
    recomendation = get_recomendation(corpus)

    query = st.text_input("Introduce your query:")

    id_search = []
    title_search = []

    final_query,q = process_query(query,corpus)        
    founded_docs = search_documents(final_query,q,corpus)
    for doc in founded_docs:
        id_search.append(int(doc.document.id))
        title_search.append(doc.document.title)
    if query != "":
        searched = {'Id':id_search,'Title':title_search}
        df = pd.DataFrame(searched)
        st.table(df.set_index(df.columns[0]))
        update_recomendation(corpus,founded_docs)
        recomendation = get_recomendation(corpus)

    with st.expander("Recomendations"):
        id_recom = []
        title_recom = []
        for d in [doc for doc,_ in recomendation]:
            id_recom.append(int(d.id))
            title_recom.append(d.title)
        recom = {'Id':id_recom,'Title':title_recom}
        df_r = pd.DataFrame(recom)
        st.table(df_r.set_index(df_r[0]))
f()