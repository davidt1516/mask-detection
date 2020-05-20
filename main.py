import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow('Detector de rostros', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            quit()
            cv.destroyAllWindows()
            break

video.release()
