import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk, Image
import cv2
from ultralytics import YOLO
import pytesseract
import os

# Tesseract yolu
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Modeli yükle
try:
    model = YOLO("runs/detect/train12/weights/best.pt")
except Exception as e:
    messagebox.showerror("Hata", f"Model yüklenemedi: {str(e)}")

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        # Ana pencere ayarları
        self.root.title("🚗 Plaka Tanıma Sistemi")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)
        
        # Ana stil konfigürasyonu
        style = ttk.Style()
        style.theme_use('clam')
        
        # Başlık çerçevesi
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill="x", padx=10, pady=(10, 0))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🚗 Plaka Tanıma Sistemi", 
            font=("Arial", 24, "bold"),
            fg="white", 
            bg="#2c3e50"
        )
        title_label.pack(expand=True)
        
        # Alt başlık
        subtitle_label = tk.Label(
            self.root,
            text="Görsel yükleyerek plaka tanıma işlemi yapın",
            font=("Arial", 12),
            fg="#7f8c8d",
            bg="#f0f0f0"
        )
        subtitle_label.pack(pady=(10, 20))
        
        # Ana içerik çerçevesi
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(expand=True, fill="both", padx=20)
        
        # Buton çerçevesi
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # Yükle butonu
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Görsel Yükle ve Tanı",
            command=self.predict_plate,
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            relief="flat",
            padx=30,
            pady=15,
            cursor="hand2",
            activebackground="#2980b9",
            activeforeground="white"
        )
        self.load_btn.pack()
        
        # Görsel önizleme çerçevesi
        preview_frame = tk.LabelFrame(
            main_frame,
            text="Plaka Önizleme",
            font=("Arial", 12, "bold"),
            fg="#2c3e50",
            bg="#f0f0f0",
            relief="ridge",
            bd=2
        )
        preview_frame.pack(pady=20, padx=10, fill="x")
        
        # Görsel paneli
        self.panel = tk.Label(
            preview_frame,
            text="Henüz görsel yüklenmedi",
            font=("Arial", 12),
            fg="#95a5a6",
            bg="white",
            width=40,
            height=8,
            relief="sunken",
            bd=2
        )
        self.panel.pack(pady=20, padx=20)
        
        # Sonuç çerçevesi
        result_frame = tk.LabelFrame(
            main_frame,
            text="Tanıma Sonucu",
            font=("Arial", 12, "bold"),
            fg="#2c3e50",
            bg="#f0f0f0",
            relief="ridge",
            bd=2
        )
        result_frame.pack(pady=(0, 20), padx=10, fill="x")
        
        # Sonuç etiketi
        self.result_label = tk.Label(
            result_frame,
            text="Plaka sonucu burada gösterilecek",
            font=("Arial", 14),
            fg="#34495e",
            bg="white",
            height=3,
            relief="sunken",
            bd=1
        )
        self.result_label.pack(pady=20, padx=20, fill="x")
        
        # Durum çubuğu
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Hazır",
            font=("Arial", 10),
            fg="white",
            bg="#34495e",
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        # Hover efektleri
        self.setup_hover_effects()
    
    def setup_hover_effects(self):
        def on_enter(e):
            self.load_btn.config(bg="#2980b9")
        
        def on_leave(e):
            self.load_btn.config(bg="#3498db")
        
        self.load_btn.bind("<Enter>", on_enter)
        self.load_btn.bind("<Leave>", on_leave)
    
    def predict_plate(self):
        try:
            # Durum güncelle
            self.status_label.config(text="Görsel seçiliyor...")
            self.root.update()
            
            file_path = filedialog.askopenfilename(
                title="Plaka görseli seçin",
                filetypes=[
                    ("Tüm desteklenen formatlar", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("JPEG dosyaları", "*.jpg *.jpeg"),
                    ("PNG dosyaları", "*.png"),
                    ("Tüm dosyalar", "*.*")
                ]
            )
            
            if not file_path:
                self.status_label.config(text="İşlem iptal edildi")
                return
            
            # Dosya kontrolü
            if not os.path.exists(file_path):
                messagebox.showerror("Hata", "Seçilen dosya bulunamadı!")
                self.status_label.config(text="Hata: Dosya bulunamadı")
                return
            
            # Durum güncelle
            self.status_label.config(text="Plaka tanınıyor...")
            self.root.update()
            
            # Görsel yükle
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Hata", "Görsel dosyası okunamadı!")
                self.status_label.config(text="Hata: Görsel okunamadı")
                return
            
            # Model tahmin
            results = model.predict(source=img, conf=0.5, verbose=False)
            boxes = results[0].boxes
            
            if not boxes:
                self.result_label.config(
                    text="❌ Plaka bulunamadı\nLütfen daha net bir görsel deneyin",
                    fg="#e74c3c"
                )
                self.panel.config(
                    text="Plaka tespit edilemedi",
                    image="",
                    compound="center"
                )
                self.panel.image = None
                self.status_label.config(text="Plaka bulunamadı")
                return
            
            # İlk tespit edilen plakayı işle
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]
            
            # OCR için görsel ön işleme
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Kontrast artırma
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Eşikleme
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Sonucu temizle
            plate_text = ''.join(c for c in text if c.isalnum()).upper()
            
            if plate_text:
                self.result_label.config(
                    text=f"✅ Tespit Edilen Plaka:\n{plate_text}",
                    fg="#27ae60"
                )
                self.status_label.config(text=f"Başarılı: {plate_text}")
            else:
                self.result_label.config(
                    text="⚠️ Plaka tespit edildi ancak\nmetin okunamadı",
                    fg="#f39c12"
                )
                self.status_label.config(text="Metin okunamadı")
            
            # Kesilen görseli göster
            img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Görsel boyutunu ayarla
            display_size = (300, 100)
            img_pil = img_pil.resize(display_size, Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img_pil)
            self.panel.config(image=img_tk, text="")
            self.panel.image = img_tk
            
        except Exception as e:
            error_msg = f"Bir hata oluştu: {str(e)}"
            messagebox.showerror("Hata", error_msg)
            self.result_label.config(text="❌ İşlem başarısız", fg="#e74c3c")
            self.status_label.config(text="Hata oluştu")

# Uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = PlateRecognitionApp(root)
    root.mainloop()