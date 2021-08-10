from flask_wtf import FlaskForm
from wtforms import TextField, TextAreaField, SubmitField, StringField
from wtforms.validators import Required, DataRequired, InputRequired

class ParsForm(FlaskForm):
    text = TextAreaField(validators=[DataRequired()])
    submit = SubmitField("Распознать")