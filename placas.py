import cv2
import numpy as np
import easyocr
import datetime
import os

# RTSP do DVR (troque pelos seus dados)
RTSP_URL = "rtsp://admin:186515Bb@192.168.100.4:554/cam/realmonitor?channel=1&subtype=0"

# Inicializa leitor de OCR
ocr = easyocr.Reader(['en'])

# Cria pasta para salvar placas
os.makedirs("placas_detectadas", exist_ok=True)

# Inicializa o vídeo
cap = cv2.VideoCapture(RTSP_URL)

# Função para melhorar imagem
def melhorar_imagem(frame):
    frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
    return frame

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar vídeo")
        break

    # Melhorar imagem com IA leve
    frame_melhorado = melhorar_imagem(frame.copy())

    # Conversão para grayscale
    gray = cv2.cvtColor(frame_melhorado, cv2.COLOR_BGR2GRAY)

    # Detecta textos (incluindo placas)
    results = ocr.readtext(gray)

    for (bbox, text, prob) in results:
        if len(text) >= 5 and prob > 0.4:
            # Desenhar a detecção
            pts = np.array(bbox).astype(int)
            cv2.polylines(frame_melhorado, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame_melhorado, text, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Salvar imagem com data/hora
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"placas_detectadas/placa_{text}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[+] Placa detectada: {text}")

    # Exibe ao vivo
    cv2.imshow("Camera com OCR", frame_melhorado)
    
    # Encerra com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
