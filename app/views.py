from flask import render_template, redirect, request, url_for
from app import app
from .forms import ParsForm



@app.route("/", methods=["GET", "POST"])
def index():
    form = ParsForm()
    messages = None
    if form.validate_on_submit():
        messages = form.text.data
        form.text.data = ''

    return render_template("index.html", form=form, messages=messages)


@app.route('/analytics',  methods=['GET', 'POST'])
def analytics():
    return render_template('analytics.html')

@app.route('/info',  methods=['GET', 'POST'])
def information():
    return render_template('info.html')