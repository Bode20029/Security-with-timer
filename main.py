import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_start_time = None
SECONDS_TO_DETECT = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5) 
    
    if len(faces) + len(bodies) > 0:
        if not detection:
            detection = True
            detection_start_time = time.time()
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started recording!")
    else:
        if detection and time.time() - detection_start_time >= SECONDS_TO_DETECT:
            detection = False
            out.release()
            print('Stop Recording!')
    
    if detection:
        out.write(frame)

    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()
