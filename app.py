from flask import Flask, render_template, Response
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

#camera = cv2.VideoCapture(0)

#0은 전면, 1은 후면
cam = cv2.VideoCapture(cv2.CAP_DSHOW+1)

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
            ret, buffer = cv2.imencode('.jpg', frame)
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