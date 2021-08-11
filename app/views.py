from flask import render_template, redirect, request, url_for
from app import app
from .forms import ParsForm

import spacy
from spacy import displacy
from flaskext.markdown import Markdown

Markdown(app)
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

nlp = spacy.load("ru_core_news_lg")


@app.route("/", methods=["GET", "POST"])
def index():
    form = ParsForm()
    text = None
    result = None
    options = {'compact': True, 'font': "Tahoma"}
    if form.validate_on_submit():
        text = form.text.data
        doc = nlp(text)
        spans = list(doc.sents)
        html = displacy.render(spans, style="dep", options=options, page=True)
        html = html.replace("\n\n", "\n")
        result = HTML_WRAPPER.format(html)

    return render_template("index.html", form=form, text=text, result=result)


@app.route('/analytics',  methods=['GET', 'POST'])
def analytics():
    return render_template('analytics.html')

@app.route('/info',  methods=['GET', 'POST'])
def information():
    return render_template('info.html')