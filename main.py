import mediapipe as mp
import cv2
import numpy as np
import gaze

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

# camera stream:
cap = cv2.VideoCapture(0)  # chose camera index
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        image=cv2.flip(image,1)
        image = draw_grid(image,(3,3))
        if not success:  # no frame input
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        landmark_points=results.multi_face_landmarks
        frame_h,frame_w,_ =image.shape
        if landmark_points:
            landmarks=landmark_points[0].landmark
            for landmark in landmarks:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(image,(x,y),3,(0,255,0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks)
            gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation

        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()