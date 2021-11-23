from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
from google.colab.patches import cv2_imshow
import numpy as np
import cv2
import imutils
from skimage import io

imgUrl = "https://github.com/juanDjulioS/Escaner-de-documentos/blob/main/prueba3.jpg?raw=true"
img = io.imread(imgUrl)
originalImg = img.copy()
ratio = img.shape[0] / 500.0
img = imutils.resize(img, height = 500)
 # Imagen en escala de grises
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 # Realizar filtrado gaussiano
grayImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
 # Realizar procesamiento de detección de bordes
imgEdge = cv2.Canny(grayImg, 50, 200)

 # Mostrar y guardar los resultados
print("Deteccuón de bordes")
cv2_imshow(img)
cv2_imshow(imgEdge) 

contours = cv2.findContours(imgEdge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]


for i in contours:
  # Se aproixman los contornos por medio de  polígonos
  perimeter = cv2.arcLength(i, True)
  approx = cv2.approxPolyDP(i, 0.04 * perimeter, True)
  if len(approx) == 4:
    screenCnt = approx
    break
    
 # Mostramos los resultados
print("STEP 2: Encontramos los contornos en el papel")
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv2_imshow(img)

# Utilice puntos de coordenadas para la transformación de coordenadas
warped = four_point_transform(originalImg, screenCnt.reshape(4, 2) * ratio)

 # Convierte el resultado transformado a valor gris
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
 # Obtenga el umbral del área local
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
 # Realizar procesamiento de binarización
warped = (warped > T).astype("uint8") * 255

 # Mostrar y guardar los resultados
print("STEP 3: Aplicar transformación perspectiva")
cv2_imshow(originalImg)
cv2_imshow(warped)
