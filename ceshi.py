from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import dlib
from imutils import face_utils
import face_recognition
import sys




def getFaceEncoding(src):
    image = face_recognition.load_image_file(src)  # 加载人脸图片
    # 获取图片人脸定位[(top,right,bottom,left )]
    face_locations = face_recognition.face_locations(image)
    #img_ = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
    #img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    # display(img_)
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]  # 对人脸图片进行编码
    return face_encoding


def simcos(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)
    sim = 1.0 / (1.0 + dist)  #
    return sim


# 提供对外比对的接口 返回比对的相似度
def comparison(face_src1, face_src2):
    xl1 = getFaceEncoding(face_src1)
    xl2 = getFaceEncoding(face_src2)
    value = simcos(xl1, xl2)
    # print(value)
    return  value


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
ap = argparse.ArgumentParser()
ap.add_argument("-m",
                "--model",
                type=str,
                required=True,
                help="path to trained model")
ap.add_argument("-l",
                "--le",
                type=str,
                required=True,
                help="path to label encoder")
ap.add_argument("-d",
                "--detector",
                type=str,
                required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(
    [args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # 从线程视频流中抓取帧，并将其调整为最大宽度为600像素
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 确保检测到的边界框不在框架的尺寸范围内
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # 提取面部ROI，进行与之前训练数据相同的预处理
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # 通过训练过的活体检测模型来检测人脸ROI是real还是fake
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]


            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, 'zsq', (startX, startY - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, 'v_smile', (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 125), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255),
                          2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.imwrite('out.jpg', frame)
        break
src = r'zsq.jpg'

value = comparison('out.jpg', src)

if value >= 0.7:
    print('人脸认证成功')
elif value < 0.7:
    print('人脸认证失败')

#
cv2.destroyAllWindows()
vs.stop()


