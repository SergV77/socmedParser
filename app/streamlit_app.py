import spacy_streamlit
import spacy
from spacy_streamlit import process_text
from spacy_streamlit import visualize_similarity
import streamlit as st
from spacy_streamlit import load_model
from visual import *
import ru_core_news_lg

# nlp = spacy.load("ru_core_news_lg")
spacy_model = st.sidebar.selectbox("Model name", ["ru_core_news_lg"])
nlp = load_model(spacy_model)
# models = ["ru_core_news_lg"]
default_text = "Шаблонный текст для примера."

st.title("Синтаксический разбор")

spacy_streamlit.visualize(nlp, default_text)


# text = st.text_area("Текст для анализа", default_text, height=200)
# doc = spacy_streamlit.process_text(*models, text)


#
# tmp_text = 'Анестезиологические и респираторные медицинские изделия. Ортопедические медицинские изделия.'
# doc = nlp(tmp_text)
# spans = doc.sents
# doc.user_data["title"] = "Это заголовок"
# options = {'compact': True, 'font': "Tahoma"}
# # displacy.serve(spans, style='dep', options=options, host='localhost')
# displacy.render([spans], style="dep", page=True)