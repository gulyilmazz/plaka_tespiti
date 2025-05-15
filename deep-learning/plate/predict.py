from ultralytics import YOLO 
import cv2 
import matplotlib.pyplot as plt 
import pytesseract 

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Modeli yükle
model_path = "runs/detect/train5/weights/best.pt"
model = YOLO(model_path)

# Görseli yükle
img_path = "dataset/images/1921.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Resim yüklenemedi. Yol yanlış mı? → {img_path}")
    exit()

# Tahmin yap
results = model.predict(source=img, conf=0.5, verbose=False)

boxes = results[0].boxes

if not boxes:
    print("Model plaka tespit edemedi.")
    exit()

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    h, w, _ = img.shape

    pad = 30
    x1p = max(x1 - pad, 0)
    y1p = max(y1 - pad, 0)
    x2p = min(x2 + pad, w)
    y2p = min(y2 + pad, h)

    cropped = img[y1p:y2p, x1p:x2p]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    text = pytesseract.image_to_string(thresh, config="--psm 7")
    print("OCR Sonucu:", text.strip())

    # Görsel göster
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title(f"Tespit Edilen Plaka: {text.strip()}")
    plt.axis('off')
    plt.show()