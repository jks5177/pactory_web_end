from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'CHECKERS'

    userid = db.Co
    user_name = db.Column(db.Integer, primary_key = True)
    user_company = db.Column(db.String(32))
   
