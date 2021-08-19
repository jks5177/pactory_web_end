from flask import Flask, render_template, Response
from flask import Flask, url_for, render_template, request, redirect, session
#from models import db
#from models import Fcuser
#from flask_wtf.csrf import CSRFProtect
#from forms import ResiterForm,LoginForm
import os
from importlib import import_module
import cv2
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
    dst = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    # threshold를 이용하여 binary image로 변환
    ret, thresh = cv2.threshold(edged, 127, 255, 0)

    # contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
    # hierachy는 contours line의 계층 구조
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # dst = cv2.drawContours(dst, contours, -1, (0, 255, 0), 3)
    #
    # return dst, (0,0,0,0)

    # print(len(contours))
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    # print(len(cnts))
    # try :
    # cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    dst = cv2.drawContours(dst, cnts, 0, (0, 255, 0), 3)
    try :
        corner = list(range(4))  # 사각형의 꼭짓점 저장

        # 시작 위치
        st = cnts[0][0]

        # 시작 위치의 대각선 꼭짓점 찾기
        fMaxDist = 0.0
        for i in range(len(cnts[0])):
            pt = cnts[0][i]
            fDist = ((st[0][0] - pt[0][0]) ** 2 + (st[0][1] - pt[0][1]) ** 2) ** 0.5

            if fDist > fMaxDist:
                corner[0] = pt
                fMaxDist = fDist

        # 이전 corner의 꼭짓점에서 대각선 꼭짓점 찾기
        fMaxDist = 0.0
        for i in range(len(cnts[0])):
            pt = cnts[0][i]
            fDist = ((corner[0][0][0] - pt[0][0]) ** 2 + (corner[0][0][1] - pt[0][1]) ** 2) ** 0.5

            if fDist > fMaxDist:
                corner[1] = pt
                fMaxDist = fDist

        # 이전 2개의 꼭짓점과 가장 먼 꼭짓점 찾기
        sumMaxDist = 0.0
        for i in range(len(cnts[0])):
            pt = cnts[0][i]
            fDist = ((st[0][0] - pt[0][0]) ** 2 + (st[0][1] - pt[0][1]) ** 2) ** 0.5
            sDist = ((corner[0][0][0] - pt[0][0]) ** 2 + (corner[0][0][1] - pt[0][1]) ** 2) ** 0.5

            sumDist = fDist + sDist

            if sumDist > sumMaxDist:
                corner[2] = pt
                sumMaxDist = sumDist

        # 3번째 꼭짓점과 가장 멀리 떨어진, 즉 대각선의 꼭짓점 찾기
        fMaxDist = 0.0
        for i in range(len(cnts[0])):
            pt = cnts[0][i]
            fDist = ((corner[2][0][0] - pt[0][0]) ** 2 + (corner[2][0][1] - pt[0][1]) ** 2) ** 0.5

            if fDist > fMaxDist:
                corner[3] = pt
                fMaxDist = fDist

        # 좌표 표시하기
        cv2.circle(dst, tuple(corner[0][0]), 5, (0, 0, 255), 2)
        cv2.circle(dst, tuple(corner[1][0]), 5, (0, 0, 255), 2)
        cv2.circle(dst, tuple(corner[2][0]), 5, (0, 0, 255), 2)
        cv2.circle(dst, tuple(corner[3][0]), 5, (0, 0, 255), 2)

        corner_ = np.array(corner)
        corner_ = corner_.reshape(-1, 2)
        corner_ = corner_[corner_[:, 0].argsort()]

        L = corner_[:2]
        R = corner_[2:4]
        L = L[L[:, 1].argsort()]
        R = R[R[:, 1].argsort()]
        LU = L[0]
        LD = L[1]
        RU = R[0]
        RD = R[1]

        return dst, (LU, LD, RU, RD)
        # return dst, (0, 0, 0, 0)

    except :
        return img, (0,0,0,0)

#camera = cv2.VideoCapture(0)

#0은 전면, 1은 후면
cam = cv2.VideoCapture(cv2.CAP_DSHOW+1)

#사용자 등록 페이지
@app.route('/register', methods=['GET','POST'])
def register():
	if request.method =='GET':
		return render_template("register.html")
	else:
		user_name = request.form.get('user_name')
		user_company = request.form.get('user_company')

		if not (user_name and user_company):
			return "모두 입력해주세요"
		else:
			user = User()
			user.user_name = user_name
			user.user_company = user_company
			db.session.add(user)
			db.session.commit()
			return "회원가입 완료"
		return redirect('/')
#로그인 페이지
# login 페이지 접속(GET) 처리와, "action=/login" 처리(POST)처리 모두 정의
@app.route('/login', methods=['GET', 'POST'])	
def login():
	if request.method=='GET':
		return render_template('login.html')
	else:
		user_company = request.form['user_company']
		user_name = request.form['user_name']
		try:
			data = User.query.filter_by(user_company=user_company, user_name=user_name).first()	# ID/PW 조회Query 실행
			if data is not None:	# 쿼리 데이터가 존재하면
				session['user_company'] = user_company	# userid를 session에 저장한다.
				return redirect('/')
			else:
				return 'Dont Login'	# 쿼리 데이터가 없으면 출력
		except:
			return "dont login"	# 예외 상황 발생 시 출력

@app.route('/logout', methods=['GET'])
def logout():
	session.pop('userid', None)
	return redirect('/') 
    

#메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

#OCR 분석 페이지
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()  # read the camera frame
        
        if (not ret):
            break
        else:
            frame_, (LU, LD, RU, RD) = preprocess(frame)
            ret, buffer = cv2.imencode('.jpg', frame_)
            frame = buffer.tobytes()
            yield (b'--frame`\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   # concat frame one by one and show result
            
@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
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

