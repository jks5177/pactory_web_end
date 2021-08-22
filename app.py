import flask
from flask import Flask, render_template, Response
from flask import Flask, url_for, render_template, request, redirect, session
# from model import db
# from model import Fcuser
# from flask_wtf.csrf import CSRFProtect
# from form import ResiterForm,LoginForm
import os
from importlib import import_module
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import time
import datetime
import sys
from flask import Flask
from flask import Flask, render_template
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
# 사각형 이미지 detection
def preprocess(img) :
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

    img = cv2.GaussianBlur(dst, (0,0), 5) # 이미지 blur 처리

    roi = dst[LU[1]:LD[1], LU[0]:RU[0]] # 관심 영역 설정

    img[LU[1]:LD[1], LU[0]:RU[0]] = roi # 해당 영역 roi로 변환

    for i in [[LU, LD], [LU, RU], [RU, RD], [LD, RD]] :
        length = abs(i[0][0]-i[1][0])+abs(i[0][1]-i[1][1])
        # cv2.line(img, i[0], i[1], (0,255,0), 3)
        for j in range(0,100,2) :
            if i[0][0] == i[1][0] :
                # print((i[0][0]+j*int(0.1*length), i[0][1]),(i[1][0]+int(j*0.1*length), i[1][1]))
                cv2.line(img, (i[0][0], i[0][1]+int(j*0.01*length)),(i[0][0], i[0][1]+int((j+1)*0.01*length)), (112,230,230), 2)
            elif i[0][1] == i[1][1] :
                cv2.line(img, (i[0][0]+int(j*0.01*length), i[0][1]),(i[0][0]+int((j+1)*0.01*length), i[0][1]), (112,230,230), 2)
    return img


#camera = cv2.VideoCapture(0)

#0은 전면, 1은 후면
cam = cv2.VideoCapture(cv2.CAP_DSHOW+2)
# cam = cv2.VideoCapture(cv2.CAP_DSHOW)

# #사용자 등록 페이지
# @app.route('/register', methods=['GET','POST'])
# def register():
# 	if request.method =='GET':
# 		return render_template("register.html")
# 	else:
# 		user_name = request.form.get('user_name')
# 		user_company = request.form.get('user_company')
#
# 		if not (user_name and user_company):
# 			return "모두 입력해주세요"
# 		else:
# 			user = User()
# 			user.user_name = user_name
# 			user.user_company = user_company
# 			db.session.add(user)
# 			db.session.commit()
# 			return "회원가입 완료"
# 		return redirect('/')
#
# #로그인 페이지
# # login 페이지 접속(GET) 처리와, "action=/login" 처리(POST)처리 모두 정의
# @app.route('/login', methods=['GET', 'POST'])
# def login():
# 	if request.method=='GET':
# 		return render_template('login.html')
# 	else:
# 		user_company = request.form['user_company']
# 		user_name = request.form['user_name']
# 		try:
# 			data = User.query.filter_by(user_company=user_company, user_name=user_name).first()	# ID/PW 조회Query 실행
# 			if data is not None:	# 쿼리 데이터가 존재하면
# 				session['user_company'] = user_company	# userid를 session에 저장한다.
# 				return redirect('/')
# 			else:
# 				return 'Dont Login'	# 쿼리 데이터가 없으면 출력
# 		except:
# 			return "dont login"	# 예외 상황 발생 시 출력
#
# @app.route('/logout', methods=['GET'])
# def logout():
# 	session.pop('userid', None)
# 	return redirect('/')
    

#메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

#OCR 분석 페이지
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()  # read the camera frame
        
        if (not ret): # 프레임이 없을 경우
            break
        else: # 프레임이 있을 경우
            frame_ = preprocess(frame) # 전처리 진행(가이드 라인 생성)
            ret, buffer = cv2.imencode('.jpg', frame_) # frame을 jpg 파일로 인코딩 진행
            frame = buffer.tobytes() # 데이터 전송을 위해 바이트 형으로 변환
            yield (b'--frame`\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # html 해당 위치로 frmae 전송
                   # concat frame one by one and show result
            
@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/camera_result')
def camera_result():
    return render_template('camera_result.html')

@app.route('/picture')
def taking_picture(): # 사진을 저장하는 페이지
    ret, frame = cam.read()  # read the camera frame

    d = pytesseract.image_to_data(frame, output_type=Output.DICT) # pytesseract로 ORC 검사

    print(d['text'])
    for i in range(len(d['text'])):
        # print(i)
        text = d['text'][i].strip()
        if (d['text'][i].startswith('KM') or d['text'][i].startswith('KN')) and len(d['text'][i]) > 15:
            # text가 KM 또는 KN으로 시작하고 text의 길이가 15 이상인 것만 추출
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            frame = cv2.rectangle(frame, (x-5, y-3), (x + w+10, y + h+6), (0, 255, 0), 2) # 해당 위치에 bounding box 생성
            text = d['text'][i]

            # 파일 이름 생성
            now = time.localtime()
            s = '%04d-%02d-%02d-%02d-%02d-%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            file_path = 'image/check_'+s+'.jpg'

            # 파일 저장 시간.jpg는 매번 바뀌는 이미지, check.jpg는 저장할 수 있는 이미지
            cv2.imwrite('static/'+file_path,frame)
            cv2.imwrite('static/image/check.jpg', frame)

            # text 파일 생성, 나중에 파일 이름으로 사용될 OCR 결과 값
            f = open("static/image/check.txt", 'w')
            f.write(d['text'][i])
            f.close()

            time.sleep(1)

            return render_template('camera_result.html', image_file=file_path, text=text)
            # return flask.redirect(flask.url_for('camera_result', image_file='image/check.jpg', text=text))

    return flask.redirect(flask.url_for('camera'))

@app.route('/del')
def del_img(): # 이미지 삭제
    import os
    path_dir = 'static/image'
    file_list = os.listdir(path_dir)
    for filename in file_list :
        file_path = path_dir + '/' + filename
        os.remove(file_path)
    return flask.redirect(flask.url_for('camera'))

@app.route('/save', methods=['GET', 'POST'])
def save_img(): # 이미지 저장
    import os
    img = cv2.imread('static/image/check.jpg')

    f = open("static/image/check.txt", 'r')
    text = f.read()
    f.close()
    print(text)

    file_path = 'static/complete/' + str(text) + '.jpg'
    cv2.imwrite(file_path, img)

    # temp 폴더 내 파일 제거
    path_dir = 'static/image'
    file_list = os.listdir(path_dir)
    for filename in file_list:
        file_path = path_dir + '/' + filename
        os.remove(file_path)

    return flask.redirect(flask.url_for('camera'))


@app.route('/video_feed')
def video_feed(): # 프레임을 실시간으로 전송해주는 페이지
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#선박 접안예측 페이지 
@app.route('/port')
def port():
    return render_template('port.html')

#실시간 정보공유 페이지
@app.route('/total')
def total():
    return render_template('total.html')

if __name__ == '__main__':
    app.run('localhost', 4997)

