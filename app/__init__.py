import os.path
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy



basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "datasqlite")
app.config["SQLALCHEMY_COMMIT_ON_TEARDOWN"] = True
app.config.from_object("config")

db = SQLAlchemy(app)
bootstrap = Bootstrap(app)

from app import views


