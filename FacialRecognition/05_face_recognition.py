# 複数の顔を認識して，対人感情を書き出す。
#
# 20240610：制作

import cv2
import numpy as np
import os 

#大域変数
g_angry=0
g_disgust = 0
g_fear = 0
g_sadness = 0
g_happiness = 0
g_surprise = 0


#'opencv-contrib-python'パッケージをインストールしていることを確認してください
#インストールするには次のコマンドを使用します: pip install opencv-contrib-python

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# IDカウンターを初期化
id = 0

# IDに関連する名前: 例 ==> Marcelo: id=1, など
names = ['None', 'KIN TAKUU', 'YAKUSHIMARU', 'PUTIN'] 

#ここから感情を設定する
emotions = {
	'KIN TAKUU': 'SURPRISE',
	'YAKUSHIMARU': 'HAPPY',
	'PUTIN': 'ANGRY'
}
#ここまで感情を設定する

# リアルタイムビデオキャプチャを初期化して開始
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


#ここから感情値の書き込み
def emotion_write():
	global g_angry  # ここで大域変数を明示
	global g_disgust
	global g_fear
	global g_sadness
	global g_happiness
	global g_surprise

#本当は，emotion_nowをコピーして，読み込む＋上書きする。
#	shutil.copy('./emotion/temp/temp_emotion_now.csv', './data/emotion_now.csv')
#	shutil.copy('./emotion/temp/temp_emotion_now.csv', './emotion/temp/emotion_now.csv')

	try:
		with open('./temp_emotion_now.csv', 'w') as fp_temp_emotion_now:
			fp_temp_emotion_now.write("時間, 怒り, 嫌悪, 恐れ, 悲しみ, 幸福, 驚き\n")
			fp_temp_emotion_now.write(f"0.0, {g_angry}, {g_disgust}, {g_fear}, {g_sadness}, {g_happiness}, {g_surprise}\n")
	except IOError:
		print("1. ./temp_emotion_now.csvファイルオープンに失敗しました。")

#	shutil.copy('./emotion/temp/temp_emotion_now.csv', './data/emotion_now.csv')
#	shutil.copy('./emotion/temp/temp_emotion_now.csv', './emotion/temp/emotion_now.csv')
#ここまで感情値の書き込み


#ここからmain
while True:
	ret, img = cam.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(int(minW), int(minH)),
	)

	#感情値を初期化
	g_angry=0
	g_disgust=0
	g_fear=0
	g_sadness=0
	g_happiness=0
	g_surprise=0

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

		id, confidence = recognizer.predict(gray[y:y + h, x: x + w])

		# 信頼度が100未満かどうか確認 ==> "0"は完全一致
		if confidence < 100:
			name = names[id]
			confidence_text = "  {0}%".format(round(100 - confidence))
			emotion = emotions.get(name, "No emotion detected") #ここで感情を表示する
			
			# 感情値の設定
			# 今は顔を認識できたときだけemotion_nowを書き出している。本当はemotion_nowをLEIからコピーして上書き必要。
			confidence_flag=1
			if name == 'KIN TAKUU':
				g_surprise = 1.0
			if name == 'YAKUSHIMARU':
				g_happiness = 1.0
			if name == 'PUTIN':
				g_angry = 1.0
		else:
			confidence_flag=0
			name = "unknown"
			confidence_text = "  {0}%".format(round(100 - confidence))
			emotion = "No emotion detected" #ここで感情を表示する
		
		if confidence_flag==1: #感情値の書き込み
			emotion_write()
			confidence_flag=0
			
		cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
		cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
		cv2.putText(img, str(emotion), (x + 5, y + h + 20), font, 1, (255, 255, 0), 1)

	cv2.imshow('camera', img)

	k = cv2.waitKey(10) & 0xff  # 'ESC'キーを押すとビデオを終了します
	if k == 27:
		break

# 終了動作
print("\n [INFO] プログラムを終了します")
cam.release()
cv2.destroyAllWindows()
#ここまでmain

