from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

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

migrate = Migrate(app, db)
# ORM模型映射成表的三步
# 1.flask db init:这步只需要执行一次
# 2.flask db migrate: 识别ORM模型的改变，生成迁移脚本#
# 3.flask db upgrade: 运行迁移脚本，同步到数据库中

'''
with app.app_context():
    with db.engine.connect() as conn:
        rs = conn.execute("SELECT 1")
        print(rs.fetchone())  # (1,)
'''


# 必须写db.model使用ORM模型,ORM可以防止SQL注入，注意区分大小写
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 添加字段,整型，主键，自增长
    username = db.Column(db.String(20), nullable=False)  # 最长100字符，不能为空
    password = db.Column(db.String(20), nullable=False)


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
