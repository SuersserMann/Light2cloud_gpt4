import os
import random
from flask import Flask, render_template, redirect, url_for, flash, abort, session, request, jsonify, send_from_directory
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, ValidationError, RadioField
from wtforms.validators import Email, DataRequired, Length, EqualTo
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from threading import Thread
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

app = Flask(__name__)
# MYSQL所在的主机名
HOSTNAME = "localhost"
# MYSQL监听的端口号，默认3306
PORT = 3306
# 连接MYSQL的用户名
USERNAME = "root"
# 连接MYSQL的密码
PASSWORD = "zzzz123zz"
# MYSQL上创建的数据库名称
DATABASE = "data_test"
app.config[
    'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8"
# 一定要注意即使有空格也会报错
# 在app.config中设置好连接数据库的信息
# 然后使用SQLALchemy(app)创建一个db对象中
# SQLALchemy会自动读取app.config中连接数据库的信息
db = SQLAlchemy(app)

migrate = Migrate(app,db)
#ORM模型映射成表的三步
# 1.fask db init:这步只需要执行一次
# 2.flask db migrate: 识别ORM模型的改变，生成迁移脚本#
# 3.flask db upgrade: 运行迁移脚本，同步到数据库中
'''
#测试是否连接数据库成功，成功则返回（1，）
with app.app_context():
    with db.engine.connect() as conn:
        rs = conn.execute("SELECT 1")
        print(rs.fetchone())  # (1,)
'''


# 必须写db.model使用ORM模型,ORM可以防止SQL注入，注意区分大小写
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 添加字段,整型，主键，自增长
    username = db.Column(db.String(20), nullable=False)  # 最长20字符，不能为空
    password = db.Column(db.String(20), nullable=False)
    # 配套下面的外键互相调用
    # articles = db.relationship("Articles", back_populates="author")


class Article(db.Model):
    __tablename__ = "article"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    # 添加作者的外键
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    author = db.relationship("User")
    # author = db.relationship("User",back_ref="articles")
    # 使用ref则不需要在原有的表添加，因为会总动创建articles在user,可以直接调用
    # author = db.relationship("User",back_populates="articles")
    # 给User表关系，创建外键，back_pupulates反向引用，可以通过articles，如user.articles调用article

#  等于artilce.author=User.query.get(article.author_id)


# 把数据库的表同步到数据库,有很大的局限性
'''
with app.app_context():
    db.create_all()
'''

@app.route("/", methods=["GET", "POST"])
def home():
    return 'hello'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8888')
