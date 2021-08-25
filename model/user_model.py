from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'CHECKERS'

    userid = db.Column(db.Integer, primary_key = True)
    user_name = db.Column(db.String(8))
    user_company = db.Column(db.String(32))

    def __init__(self, userid, user_name, user_company):
        self.userid = userid
        self.user_name = user_name
        self.user_company = user_company

    # def create(self, id, name, company):

   
