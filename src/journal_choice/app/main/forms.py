from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length


class UploadForm(FlaskForm):
    title = StringField('Title text', default='Your title', validators=[
        DataRequired(), Length(max=500, message="500 character limit.")])
    abstract = TextAreaField('Abstract text',  default='Your abstract', validators=[
        DataRequired(), Length(max=2000, message="2000 character limit.")])
    ref_file = FileField('References RIS file', validators=[
        FileRequired(),
        FileAllowed(['ris'], 'Use .ris extension')
    ])
    submit = SubmitField('Upload')
