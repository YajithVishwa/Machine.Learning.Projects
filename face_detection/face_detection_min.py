import cv2 
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    if results.detections:
        for faceLm in results.detections:
            bboxC = faceLm.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            score = int(faceLm.score[0] * 100)
            if score > 75:
                cv2.putText(img, f'{score}%', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(10)