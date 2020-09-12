from .db import db


class FoodImages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.Text, unique=True)
    name = db.Column(db.Text, nullable=False)
    mime_type = db.Column(db.Text, nullable=False)
