import cv2
import dlib

import pandas as pd


#cap = cv2.VideoCapture('example_.mp4') #carrega o video na variavel
cap = cv2.VideoCapture(0) #Webcam object
cap.set(cv2.CAP_PROP_FPS, 30) #seta a capitura para 30 quadros por segundo

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #preve os pontos do rosto

data = pd.DataFrame() #criar data frame vazio
linha = pd.Series() #cria vetor

def retorna_cordenadas(f): #função para tratar o frame
    f = cv2.flip(f,180)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detec = detector(clahe_image, 1) #detecta a face no quadro
    return detec,clahe_image

while True: #enquanto tiver aberto o video
    
    r, frame = cap.read() #pega prximo quadro
    cv2.imshow("image", frame) #Display the frame

    if cv2.waitKey(1) & 0xFF == ord('g'):#checar o iniciar da gravação pressionando "G"
        currentFrame = 1
        linha = pd.Series() #cria vetor
        
        while currentFrame <= 90: #coletar os dados até 3 segundo de video 90 frames/quadros
            currentFrame=currentFrame+1
            
            r, frame = cap.read() #pega prximo quadro
            cv2.imshow("image", frame) #Display the frame
            
            detections,clahe_image = retorna_cordenadas(frame) #trata o quadro e retorna as coordenadas dos rostos
            
            if detections != None:
                for k,d in enumerate(detections):
                    shape = predictor(clahe_image, d) #pega cordenadas
                for i in range(1,68): #There are 68 landmark points on each face
                    linha = linha.append(pd.Series([shape.part(i).x, shape.part(i).y]),ignore_index=True) #adiciona as cordenadas no vetor
                data = data.append(linha,ignore_index=True) #adiciona um linha com os valores do vetor
            
    cv2.imshow("image", frame) #Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
    
cap.release()
cv2.destroyAllWindows()