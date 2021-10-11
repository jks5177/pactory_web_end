import flask
# import sqlalchemy.sql.expression
from flask import Flask, render_template, Response
from flask import Flask, url_for, render_template, request, redirect, session

import os
from importlib import import_module
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import time
import datetime
import sys
from flask import Flask, render_template

import sqlalchemy
import sqlalchemy as db
from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import random

import pymysql
pymysql.install_as_MySQLdb()

from PIL import Image
import base64
from io import BytesIO

buffer = BytesIO()

import socket

import pandas as pd

# db 연동
# root:내비번
engine = create_engine("mysql://new:new@13.124.122.246:3306/loading_DB")

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# db Base 클래스 생성 => DB를 가져올 class를 생성함
Base = declarative_base()
Base.query = db_session.query_property()

# DB 가져오기
connection = engine.connect()
metadata = Base.metadata
metadata.create_all(engine)

app = Flask(__name__)

import json

with open('static/car_vin.json', 'r+', encoding='utf8') as f:
    json_data = json.load(f)


def vin_decoder(car_vin):
    decode_list = []
    if car_vin[0] in json_data['car_num0'].keys():
        decode_list.append(json_data['car_num0'][car_vin[0]])
        # print('국가 :', json_data['car_num0'][car_vin[0]])
    if car_vin[1] in json_data['car_num1'].keys():
        decode_list.append(json_data['car_num1'][car_vin[1]])
        # print('회사 :', json_data['car_num1'][car_vin[1]])
    if car_vin[2] in json_data['car_num2'].keys():
        decode_list.append(json_data['car_num2'][car_vin[2]])
        # print('차량형태 :',json_data['car_num2'][car_vin[2]])
    if car_vin[3] in json_data['car_num3'].keys():
        decode_list.append(json_data['car_num3'][car_vin[3]])
        # print('차종류 :',json_data['car_num3'][car_vin[3]])
        if type(json_data['car_num3'][car_vin[3]]) == list:
            if car_vin[4] in json_data['car_num4'].keys():
                try:
                    car = set([json_data['car_num4'][car_vin[4]]]).intersection(set(json_data['car_num3'][car_vin[3]]))
                except:
                    car = set(json_data['car_num3'][car_vin[3]]).intersection(list(json_data['car_num4'][car_vin[4]]))
                try:
                    decode_list.append(list(car)[0])
                except:
                    num = random.randrange(len(car_num3[car_vin[3]]))
                    decode_list.append(car_num3[car_vin[3]][num])
            else :
                num = random.randrange(len(car_num3[car_vin[3]]))
                decode_list.append(car_num3[car_vin[3]][num])
    if car_vin[5] in json_data['car_num5'].keys():
        decode_list.append(json_data['car_num5'][car_vin[5]])
        # print(json_data['car_num5'][car_vin[5]])
    if car_vin[6] in json_data['car_num6'].keys():
        decode_list.append(json_data['car_num6'][car_vin[6]])
        # print('차량형태:', json_data['car_num6'][car_vin[6]])
    if car_vin[7] in json_data['car_num7'].keys():
        decode_list.append(json_data['car_num7'][car_vin[7]])
        # print('차정보1 : ', json_data['car_num7'][car_vin[7]])
    if car_vin[8] in json_data['car_num8'].keys():
        decode_list.append(json_data['car_num8'][car_vin[8]])
        # print(json_data['car_num8'][car_vin[8]])
    if car_vin[9] in json_data['car_num9'].keys():
        decode_list.append(json_data['car_num9'][car_vin[9]])
        # print(json_data['car_num9'][car_vin[9]])
    if car_vin[10] in json_data['car_num10'].keys():
        decode_list.append(json_data['car_num10'][car_vin[10]])
        # print('제조공장 :',json_data['car_num10'][car_vin[10]])
    return decode_list

# 로그인 페이지
@app.route('/_')
def login_page():
    return render_template('login.html')

# 메인 페이지
@app.route('/main')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return redirect('http://localhost:4997/camera')

@app.route('/camera_result')
def camera_result():
    return render_template('camera_result.html')

@app.route('/image_send')
def image_send() :
    image_df = pd.read_sql(sql='select * from TEMP', con=engine)
    img_str = image_df['IMAGE'].values[0]
    car_vin = image_df['IMAGE_NAME'].values[0]

    img = base64.decodestring(img_str)

    im = Image.open(BytesIO(img))

    numpy_image = np.array(im)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    now = time.localtime()
    s = '%04d-%02d-%02d-%02d-%02d-%02d' % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    file_path = 'image/check_' + s + '.jpg'

    directory = 'static/image'

    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 저장 시간.jpg는 매번 바뀌는 이미지, check.jpg는 저장할 수 있는 이미지
    cv2.imwrite('static/' + file_path, opencv_image)
    cv2.imwrite('static/image/check.jpg', opencv_image)

    time.sleep(1)

    temp_table = sqlalchemy.Table('TEMP', metadata, autoload=True, autoload_with=engine)
    db_session.query(temp_table).delete()
    db_session.commit()

    return render_template('camera_result.html', image_file=file_path, text=car_vin)

@app.route('/del')
def del_img():  # 이미지 삭제
    import os
    path_dir = 'static/image'
    file_list = os.listdir(path_dir)
    for filename in file_list:
        file_path = path_dir + '/' + filename
        os.remove(file_path)
    return flask.redirect(flask.url_for('camera'))


@app.route('/save', methods=['GET', 'POST'])
def save_img():  # 이미지 저장
    import os
    img = cv2.imread('static/image/check.jpg')

    if request.method == 'POST' :
        # print("post")
        cargo_vin = request.form['vin']
        # print(cargo_vin)

    # file_path = 'static/complete/' + str(cargo_vin) + '.jpg'
    # cv2.imwrite(file_path, img)

    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    login_table = sqlalchemy.Table('LOGIN', metadata, autoload=True, autoload_with=engine)
    f_s = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][-1]

    # try:
    if f_s == '1차' :
        # db 저장
        decode_list = vin_decoder(cargo_vin)
        car_name = decode_list[4]
        print(car_name)

        now = time.localtime()
        now_time = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        phoneNum = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][0]

        table = db.Table('STORAGE', metadata, autoload=True, autoload_with=engine)
        query = db.insert(table).values(CARGO_VIN=cargo_vin,CARGO_NAME=car_name,INSPECT_TIME=now_time,IP=ip,LI_PHONENUM=phoneNum)
        result_proxy = connection.execute(query)
        result_proxy.close()

        # temp 폴더 내 파일 제거
        path_dir = 'static/image'
        file_list = os.listdir(path_dir)
        for filename in file_list:
            file_path = path_dir + '/' + filename
            os.remove(file_path)

        return flask.redirect(flask.url_for('camera'))

    else :
        im = Image.fromarray(img)
        im.save(buffer, format='jpeg')
        img_str = base64.b64encode(buffer.getvalue())

        img_df = pd.DataFrame({'IMAGE_NAME': cargo_vin, 'IMAGE': [img_str]})

        img_df.to_sql('IMAGE', con=engine, if_exists='append', index=False)

        # db 저장
        decode_list = vin_decoder(cargo_vin)
        car_name = decode_list[4]
        print(car_name)

        car_table = sqlalchemy.Table('CAR', metadata, autoload=True, autoload_with=engine)
        cargo_weight = db_session.query(car_table).filter(text("CAR_NAME=:car_name")).params(car_name=car_name).all()[0][1]

        now = time.localtime()
        now_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        phoneNum = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][0]
        deck = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][5]
        hold = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][4]
        vessel_name = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][6]

        table = db.Table('CARGO', metadata, autoload=True, autoload_with=engine)
        query = db.insert(table).values(CARGO_VIN=cargo_vin, VESSEL_NAME=vessel_name,
                                        CARGO_NAME=car_name, CARGO_WEIGHT=cargo_weight, CARGO_INSPECT_TIME=now_time,
                                        IP=ip, LI_PHONENUM=phoneNum, DECK=deck, HOLD=hold)
        result_proxy = connection.execute(query)
        result_proxy.close()

        storage_table = sqlalchemy.Table('STORAGE', metadata, autoload=True, autoload_with=engine)

        db_session.query(storage_table).filter(text("CARGO_VIN=:cargo_vin")).params(cargo_vin=cargo_vin).delete()
        db_session.commit()

        # temp 폴더 내 파일 제거
        path_dir = 'static/image'
        file_list = os.listdir(path_dir)
        for filename in file_list:
            file_path = path_dir + '/' + filename
            os.remove(file_path)

        return flask.redirect(flask.url_for('camera'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' :
        # print("post")
        user_company = request.form['user_company']
        user_name = request.form['user_name']
        user_phoneNum = request.form['user_phoneNum']
        f_s = request.form = request.form['first_second']

        if len(user_company) == 0 or len(user_name) == 0 or len(user_phoneNum) == 0 or len(f_s) == 0 :
            return flask.redirect(flask.url_for('login_page'))

        else :
            # print("else")
            try :
                print(request.environ.get('HTTP_X_REAL_IP', request.remote_addr))
                table = db.Table('LOGIN', metadata, autoload=True, autoload_with=engine)
                query = db.insert(table).values(LI_PHONENUM=user_phoneNum, LI_NAME=user_name, LI_UNLOADING=user_company, IP=request.environ.get('HTTP_X_REAL_IP', request.remote_addr), F_S=f_s)
                result_proxy = connection.execute(query)
                # print(user_phoneNum, user_name, user_company, socket.gethostbyname(socket.gethostname()))
                result_proxy.close()

                if f_s == '1차' :
                    return flask.redirect(flask.url_for('camera'))
                else:
                    return flask.redirect(flask.url_for('index'))

            except :
                return flask.redirect(flask.url_for('login_page'))

    elif request.method == 'GET' :
        # print("get")
        return flask.redirect(flask.url_for('login_page'))

@app.route('/logout')
def logout() :
    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    login_table = sqlalchemy.Table('LOGIN', metadata, autoload=True, autoload_with=engine)

    db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).delete()
    db_session.commit()

    return flask.redirect(flask.url_for('login_page'))

#실시간 정보공유 페이지
@app.route('/total')
def total():

    worker_table = sqlalchemy.Table('WORKER', metadata, autoload=True, autoload_with=engine)

    try :
        checker = db_session.query(worker_table).filter(text("WORKER_TASK='checker'")).all()[0][-1]
        driver = db_session.query(worker_table).filter(text("WORKER_TASK='drive'")).all()[0][-1]
        lashing = db_session.query(worker_table).filter(text("WORKER_TASK='lashing'")).all()[0][-1]

    except :
        checker = 0
        driver = 0
        lashing = 0

    cargo_table = sqlalchemy.Table('CARGO', metadata, autoload=True, autoload_with=engine)
    data = db_session.query(cargo_table).order_by(text("CARGO_INSPECT_TIME")).all()[:6]
    # print(data)

    date = datetime.datetime

    deck = []
    for i in range(1, 12) :
        percent = (len(db_session.query(cargo_table).filter(text("DECK=:deck_num")).params(deck_num=i).all()) / 100) * 100
        deck.append(percent)

    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    total_num = len(db_session.query(cargo_table).filter(text("IP=:ip")).params(ip=ip).all())

    schedule_table = sqlalchemy.Table('SCHEDULE', metadata, autoload=True, autoload_with=engine)
    # print(deck)
    try :
        import_time = []
        export_time = []
        vessel_name = []
        schedule_list = db_session.query(schedule_table).filter(text("SCHEDULE_EXPORT>=:date")).filter(text("SCHEDULE_IMPORT<=:date")).params(date=date.now()).order_by(text("SCHEDULE_EXPORT")).all()
        for i in schedule_list :
            import_time.append(i[0])
            export_time.append(i[1])
            vessel_name.append(i[2])

        # print(schedule_list)

        limit_time = export_time[0] - date.now()
        print(limit_time)

        hour, minute, second = str(limit_time).split(':')
        print(hour, minute, second)
        return render_template('total.html', checker=checker,driver=driver,lashing=lashing,data=data,hour=hour,minute=minute,second=second,vessel_name=vessel_name,deck=deck,total_num=total_num,date=datetime.date.today())
    
    except :
        # print("except")
        hour, minute, second = 0, 0, 0
        return render_template('total.html', checker=checker,driver=driver,lashing=lashing,data=data,hour=hour,minute=minute,second=second,vessel_name=vessel_name,deck=deck,total_num=total_num, date=datetime.date.today())

@app.route('/hol_dec_send', methods=['GET','POST'])
def hol_dec_send() :
    if request.method == 'POST' :
        # print("post")

        hold = request.form['hold']
        deck = request.form['deck']

        login_table = db.Table('LOGIN', metadata, autoload=True, autoload_with=engine)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).update({'DECK':deck, 'HOLD':hold}, synchronize_session=False)
        db_session.commit()

        return flask.redirect(flask.url_for('total'))

    else :
        return flask.redirect(flask.url_for('total'))

@app.route('/vessel_send', methods=['GET','POST'])
def vessel_send() :
    if request.method == 'POST' :
        # print("post")

        vessel_name = request.form['vessel']

        login_table = db.Table('LOGIN', metadata, autoload=True, autoload_with=engine)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).update({'VESSEL_NAME':vessel_name}, synchronize_session=False)
        db_session.commit()

        return flask.redirect(flask.url_for('total'))

    else :
        return flask.redirect(flask.url_for('total'))

@app.route('/worker_send', methods=['GET', 'POST'])
def worker_send() :
    worker_table = db.Table('WORKER', metadata, autoload=True, autoload_with=engine)
    db_session.query(worker_table).delete()
    db_session.commit()

    checker_task = 'checker'
    drive_task = 'drive'
    lashing_task = 'lashing'

    if request.method == 'POST' :
        # print("post")

        checker = request.form['checker']
        drive = request.form['driver']
        lashing = request.form['lashing']

        now = time.localtime()
        today = "%04d/%02d/%02d" % (now.tm_year, now.tm_mon, now.tm_mday)

        # print(checker, drive, lashing)
        try :
            worker_table = db.Table('WORKER', metadata, autoload=True, autoload_with=engine)
            query = db.insert(worker_table).values(WORKER_TASK=checker_task, WORKER_DATE=today, WORKER_PERSONNEL=checker)
            result_proxy = connection.execute(query)
            result_proxy.close()

            query = db.insert(worker_table).values(WORKER_TASK=drive_task, WORKER_DATE=today,WORKER_PERSONNEL=drive)
            result_proxy = connection.execute(query)
            result_proxy.close()

            query = db.insert(worker_table).values(WORKER_TASK=lashing_task, WORKER_DATE=today,WORKER_PERSONNEL=lashing)
            result_proxy = connection.execute(query)
            result_proxy.close()

            return flask.redirect(flask.url_for('total'))

        except :
            return flask.redirect(flask.url_for('worker'))

    else :
        return flask.redirect(flask.url_for('worker'))

# 스케쥴 페이지
@app.route('/table')
def table():
    return render_template('table.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/cal')
def cal():
    return render_template('schedule.html')

@app.route('/schedule', methods=['GET', 'POST'])
def schedule() :
    # print("in")
    if request.method == 'POST':
        schedule_import = request.form['schedule_import']
        schedule_export = request.form['schedule_export']
        vessel_name = request.form['vessel_name']
        schedule_ton = request.form['schedule_ton']

        # print(type(schedule_ton), type(schedule_export), type(schedule_import), type(vessel_name))
        if len(schedule_import) == 0 or len(schedule_export) == 0 or len(vessel_name) == 0 :
            return flask.redirect(flask.url_for('cal'))
        else :
            # print("else")
            try :
                table = db.Table('SCHEDULE', metadata, autoload=True, autoload_with=engine)
                query = db.insert(table).values(SCHEDULE_IMPORT=schedule_import, SCHEDULE_EXPORT=schedule_export, VESSEL_NAME=vessel_name, SCHEDULE_TON=schedule_ton)
                result_proxy = connection.execute(query)
                result_proxy.close()
                return flask.redirect(flask.url_for('cal'))
            except :
                return flask.redirect(flask.url_for('cal'))
    elif request.method == 'GET' :
        # print("get")
        return flask.redirect(flask.url_for('cal'))

@app.route('/worker')
def worker():
    return render_template('worker.html')

@app.route('/deck')
def deck() :
    car_sql = 'select * from CAR'
    car_df = pd.read_sql(car_sql, con=connection)
    car_name = list(car_df['CAR_NAME'])
    print(car_name)
    # car_table = sqlalchemy.Table('cargo', metadata, autoload=True, autoload_with=engine)
    cargo_table = sqlalchemy.Table('CARGO', metadata, autoload=True, autoload_with=engine)
    # car_count_dic = {}
    car = [[0 for col in range(4)] for row in range(11)]
    for i in range(11) :
        for j in range(4) :
            for k in car_name :
                car_count = len(db_session.query(cargo_table).filter(text("DECK=:deck_num")).params(deck_num=i).filter(text("HOLD=:hold_num")).params(hold_num=j).filter(text("CARGO_NAME=:car_name")).params(car_name=k).all())
                if car_count > 0 :
                    car[i][j] = {k:car_count}

    # print(car_count_dic)
    return render_template('deck.html', car=car)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4997, debug=True)