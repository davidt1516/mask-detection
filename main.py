import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') # Carga de un filtro de cascada específico para detectar rostros
video = cv.VideoCapture(0) # Configuración para captura de video de cámara 

while video.isOpened():
    ret, frame = video.read() #Lectura de cámara
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #Conversión a escala de grises
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) #Aplicar filtro, determinando los espacios de pixeles que contienen la cara

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #Dibujar cuadrado en los rostros detectados

        cv.imshow('Detector de rostros', frame)
        if cv.waitKey(25) & 0xFF == ord('q'): #A la espera de pulsar la tecla de finalización
            #quit() <----------- Esto no tiene función práctica en el código, da error porque la función no está implementada -------------->
            cv.destroyAllWindows()
            break

video.release() #Apagado de la cámara
