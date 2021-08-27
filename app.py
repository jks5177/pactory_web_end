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

import pymysql
pymysql.install_as_MySQLdb()

import socket

import pandas as pd

# db 연동
# root:내비번
engine = create_engine("mysql://root:root@127.0.0.1:3306/loading_db")

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# db Base 클래스 생성 => DB를 가져올 class를 생성함
Base = declarative_base()
Base.query = db_session.query_property()

# DB 가져오기
connection = engine.connect()
metadata = Base.metadata
metadata.create_all(engine)

app = Flask(__name__)

'''
import cv2

# 일반적으로 웹캠 불러오기
cam = cv2.VideoCapture(0)
ret, frame = cam.read()

# 기존 방식으로 연결이 안될 경우
# 여기서 숫자 0은 웹캠의 채널 인덱스

cam = cv2.VideoCapture(cv2.CAP_DSHOW+0)
ret, frame = cam.read()
'''

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
                decode_list.append(list(car)[0])
                # print('차량상세:', list(car)[0] )
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


# 사각형 이미지 detection
def preprocess(img):
    import cv2
    import numpy as np

    r = 800.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 800)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    h, w, c = img.shape

    LU = (int(w * 0.2), int(h * 0.1))
    LD = (int(w * 0.2), int(h * 0.9))
    RU = (int(w * 0.8), int(h * 0.1))
    RD = (int(w * 0.8), int(h * 0.9))

    # print(LU, LD, RU, RD)

    dst = img.copy()

    img = cv2.GaussianBlur(dst, (0, 0), 5)  # 이미지 blur 처리

    roi = dst[LU[1]:LD[1], LU[0]:RU[0]]  # 관심 영역 설정

    img[LU[1]:LD[1], LU[0]:RU[0]] = roi  # 해당 영역 roi로 변환

    for i in [[LU, LD], [LU, RU], [RU, RD], [LD, RD]]:
        length = abs(i[0][0] - i[1][0]) + abs(i[0][1] - i[1][1])
        # cv2.line(img, i[0], i[1], (0,255,0), 3)
        for j in range(0, 100, 2):
            if i[0][0] == i[1][0]:
                # print((i[0][0]+j*int(0.1*length), i[0][1]),(i[1][0]+int(j*0.1*length), i[1][1]))
                cv2.line(img, (i[0][0], i[0][1] + int(j * 0.01 * length)),
                         (i[0][0], i[0][1] + int((j + 1) * 0.01 * length)), (42, 204, 246), 2)
            elif i[0][1] == i[1][1]:
                cv2.line(img, (i[0][0] + int(j * 0.01 * length), i[0][1]),
                         (i[0][0] + int((j + 1) * 0.01 * length), i[0][1]), (42, 204, 246), 2)
    return img

# 0은 전면, 1은 후면

# cam = cv2.VideoCapture(1)  # 아리

cam = cv2.VideoCapture(cv2.CAP_DSHOW + 1)


# 사용자 등록 페이지
@app.route('/register', methods=['GET', 'POST'])
def register():
    return redirect('/')


# 로그인 페이지
@app.route('/')
def login_page():
    return render_template('login.html')

# 메인 페이지
@app.route('/main')
def index():
    return render_template('index.html')

# OCR 분석 페이지
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()  # read the camera frame

        if (not ret):  # 프레임이 없을 경우
            break
        else:  # 프레임이 있을 경우
            frame_ = preprocess(frame)  # 전처리 진행(가이드 라인 생성)
            ret, buffer = cv2.imencode('.jpg', frame_)  # frame을 jpg 파일로 인코딩 진행
            frame = buffer.tobytes()  # 데이터 전송을 위해 바이트 형으로 변환
            yield (b'--frame`\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # html 해당 위치로 frmae 전송
            # concat frame one by one and show result


@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/camera_result')
def camera_result():
    return render_template('camera_result.html')


@app.route('/picture')
def taking_picture():  # 사진을 저장하는 페이지
    ret, frame = cam.read()  # read the camera frame

    d = pytesseract.image_to_data(frame, output_type=Output.DICT)  # pytesseract로 ORC 검사

    print(d['text'])
    for i in range(len(d['text'])):
        # print(i)
        text = d['text'][i].strip()
        if (d['text'][i].startswith('KM') or d['text'][i].startswith('KN')) and len(d['text'][i]) > 11:
            # text가 KM 또는 KN으로 시작하고 text의 길이가 15 이상인 것만 추출
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            frame = cv2.rectangle(frame, (x - 5, y - 3), (x + w + 10, y + h + 6), (0, 255, 0),
                                  2)  # 해당 위치에 bounding box 생성
            # text = d['text'][i]

            # 파일 이름 생성
            now = time.localtime()
            s = '%04d-%02d-%02d-%02d-%02d-%02d' % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            file_path = 'image/check_' + s + '.jpg'

            directory = 'static/image'

            if not os.path.exists(directory):
                os.makedirs(directory)

            # 파일 저장 시간.jpg는 매번 바뀌는 이미지, check.jpg는 저장할 수 있는 이미지
            cv2.imwrite('static/' + file_path, frame)
            cv2.imwrite('static/image/check.jpg', frame)

            time.sleep(1)

            return render_template('camera_result.html', image_file=file_path, text=text)

    return flask.redirect(flask.url_for('camera'))


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
        print(cargo_vin)

    file_path = 'static/complete/' + str(cargo_vin) + '.jpg'
    cv2.imwrite(file_path, img)

    ip = socket.gethostbyname(socket.gethostname())
    login_table = sqlalchemy.Table('login', metadata, autoload=True, autoload_with=engine)
    f_s = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][-1]

    # try:
    if f_s == '1차' :
        # db 저장
        decode_list = vin_decoder(cargo_vin)
        car_name = decode_list[4]
        print(car_name)

        car_table = sqlalchemy.Table('car', metadata, autoload=True, autoload_with=engine)
        cargo_weight = db_session.query(car_table).filter(text("CAR_NAME=:car_name")).params(car_name=car_name).all()[0][1]

        now = time.localtime()
        now_time = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        phoneNum = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][0]

        table = db.Table('storage', metadata, autoload=True, autoload_with=engine)
        query = db.insert(table).values(CARGO_VIN=cargo_vin, IMAGE_PATH=file_path,CARGO_NAME=car_name,INSPECT_TIME=now_time,IP=ip,LI_PHONENUM=phoneNum)
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
        # db 저장
        decode_list = vin_decoder(cargo_vin)
        car_name = decode_list[4]
        print(car_name)

        car_table = sqlalchemy.Table('car', metadata, autoload=True, autoload_with=engine)
        cargo_weight = \
        db_session.query(car_table).filter(text("CAR_NAME=:car_name")).params(car_name=car_name).all()[0][1]

        now = time.localtime()
        now_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        phoneNum = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][0]

        deck = db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).all()[0][5]

        table = db.Table('cargo', metadata, autoload=True, autoload_with=engine)
        query = db.insert(table).values(CARGO_VIN=cargo_vin, IMAGE_PATH=file_path, VESSEL_NAME="GLOVIS SIRIUS",
                                        CARGO_NAME=car_name, CARGO_WEIGHT=cargo_weight, CARGO_INSPECT_TIME=now_time,
                                        IP=ip, LI_PHONENUM=phoneNum, DECK=deck)
        result_proxy = connection.execute(query)
        result_proxy.close()

        storage_table = sqlalchemy.Table('storage', metadata, autoload=True, autoload_with=engine)

        db_session.query(storage_table).filter(text("CARGO_VIN=:cargo_vin")).params(cargo_vin=cargo_vin).delete()
        db_session.commit()

        # temp 폴더 내 파일 제거
        path_dir = 'static/image'
        file_list = os.listdir(path_dir)
        for filename in file_list:
            file_path = path_dir + '/' + filename
            os.remove(file_path)

        return flask.redirect(flask.url_for('camera'))

    # except:
    #     return flask.redirect(flask.url_for('camera'))


@app.route('/video_feed')
def video_feed():  # 프레임을 실시간으로 전송해주는 페이지
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 프론트 스케쥴 페이지
@app.route('/schedule')
def schedule_page():
    return render_template('schedule.html')

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
                table = db.Table('login', metadata, autoload=True, autoload_with=engine)
                query = db.insert(table).values(LI_PHONENUM=user_phoneNum, LI_NAME=user_name, LI_UNLOADING=user_company, IP=socket.gethostbyname(socket.gethostname()), F_S=f_s)
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
    ip = socket.gethostbyname(socket.gethostname())

    login_table = sqlalchemy.Table('login', metadata, autoload=True, autoload_with=engine)

    db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).delete()
    db_session.commit()

    return flask.redirect(flask.url_for('login_page'))

#실시간 정보공유 페이지
@app.route('/total')
def total():

    worker_table = sqlalchemy.Table('worker', metadata, autoload=True, autoload_with=engine)

    try :
        checker = db_session.query(worker_table).filter(text("WORKER_TASK='checker'")).all()[0][-1]
        driver = db_session.query(worker_table).filter(text("WORKER_TASK='drive'")).all()[0][-1]
        lashing = db_session.query(worker_table).filter(text("WORKER_TASK='lashing'")).all()[0][-1]

    except :
        checker = 0
        driver = 0
        lashing = 0

    cargo_table = sqlalchemy.Table('cargo', metadata, autoload=True, autoload_with=engine)
    data = db_session.query(cargo_table).all()[:6]
    # print(data)

    date = datetime.datetime

    deck = []
    for i in range(1, 12) :
        percent = (len(db_session.query(cargo_table).filter(text("DECK=:deck_num")).params(deck_num=i).all()) / 10) * 100
        deck.append(percent)

    ip = socket.gethostbyname(socket.gethostname())
    total_num = len(db_session.query(cargo_table).filter(text("IP=:ip")).params(ip=ip).all())

    schedule_table = sqlalchemy.Table('schedule', metadata, autoload=True, autoload_with=engine)
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

        limit_time = export_time[0] - date.now()

        hour, minute, second = str(limit_time).split(':')

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

        login_table = db.Table('login', metadata, autoload=True, autoload_with=engine)
        ip = socket.gethostbyname(socket.gethostname())
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

        login_table = db.Table('login', metadata, autoload=True, autoload_with=engine)
        ip = socket.gethostbyname(socket.gethostname())
        db_session.query(login_table).filter(text("IP=:ip")).params(ip=ip).update({'VESSEL_NAME':vessel_name}, synchronize_session=False)
        db_session.commit()

        return flask.redirect(flask.url_for('total'))

    else :
        return flask.redirect(flask.url_for('total'))

@app.route('/worker_send', methods=['GET', 'POST'])
def worker_send() :
    worker_table = db.Table('worker', metadata, autoload=True, autoload_with=engine)
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
            worker_table = db.Table('worker', metadata, autoload=True, autoload_with=engine)
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
def cal3():
    return render_template('schedule.html')

@app.route('/schedule', methods=['GET', 'POST'])
def schedule() :
    print("in")
    if request.method == 'POST':
        schedule_import = request.form['schedule_import']
        schedule_export = request.form['schedule_export']
        vessel_name = request.form['vessel_name']
        schedule_ton = request.form['schedule_ton']

        print(type(schedule_ton), type(schedule_export), type(schedule_import), type(vessel_name))
        if len(schedule_import) == 0 or len(schedule_export) == 0 or len(vessel_name) == 0 :
            return flask.redirect(flask.url_for('cal3'))
        else :
            print("else")
            # try :
            table = db.Table('schedule', metadata, autoload=True, autoload_with=engine)
            query = db.insert(table).values(SCHEDULE_IMPORT=schedule_import, SCHEDULE_EXPORT=schedule_export, VESSEL_NAME=vessel_name, SCHEDULE_TON=schedule_ton)
            result_proxy = connection.execute(query)
            result_proxy.close()
            return flask.redirect(flask.url_for('cal3'))
            # except:
                #     return flask.redirect(flask.url_for('schedule_page'))
    elif request.method == 'GET' :
        print("get")
        return flask.redirect(flask.url_for('cal3'))

@app.route('/worker')
def worker():
    return render_template('worker.html')


if __name__ == '__main__':
    app.run('localhost', 4997, debug=True)