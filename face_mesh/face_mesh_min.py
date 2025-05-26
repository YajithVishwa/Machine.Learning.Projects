import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))
    cv2.imshow("Image", img)
    cv2.waitKey(10)