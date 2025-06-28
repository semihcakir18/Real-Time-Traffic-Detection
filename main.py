# Gerekli kütüphaneleri içe aktarıyoruz
import cv2
from ultralytics import YOLO
import numpy as np


# --- TEMEL AYARLAR ---

# Önceden eğitilmiş YOLOv8 modelini yüklüyoruz.
model = YOLO("yolov8n.pt")

video_kaynagi = "videos/traffic.mp4"

# Video yakalama nesnesini oluşturuyoruz
cap = cv2.VideoCapture(video_kaynagi)

# Sadece belirli nesneleri tespit etmek için bir filtre listesi oluşturuyoruz.
# COCO veri setindeki tüm sınıflar için model.names'i yazdırarak görebilirsiniz.
hedef_siniflar = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "person",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
]

# --- ANA DÖNGÜ ---

while True:
    # Videodan bir kare (frame) okuyoruz
    success, frame = cap.read()

    # Eğer video bittiyse veya kare okunamadıysa döngüden çıkıyoruz
    if not success:
        print("Video bitti veya kare okunamadı.")
        break

    # Okunan kareyi YOLOv8 modeline vererek nesne tespiti yapıyoruz
    # 'stream=True' daha verimli bir işlem sağlar
    results = model.track(frame, persist=True, stream=True)

    # Her kare için nesne sayacını sıfırlıyoruz
    nesne_sayaci = {sinif: 0 for sinif in hedef_siniflar}

    # Tespit edilen sonuçlar üzerinde döngüye giriyoruz
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Tespit edilen her bir nesnenin sınıf ID'sini alıyoruz
            cls_id = int(box.cls[0])
            sinif_adi = model.names[cls_id]

            # Eğer tespit edilen sınıf, hedef sınıflarımızdan biri değilse atlıyoruz
            if sinif_adi not in hedef_siniflar:
                continue

            # Nesnenin güven skorunu alıyoruz
            guven_skoru = float(box.conf[0])

            # Güven skoru belirli bir eşiğin altındaysa (örneğin %50) atlıyoruz
            if guven_skoru < 0.5:
                continue

            # Sınıf sayacını artırıyoruz
            nesne_sayaci[sinif_adi] += 1

            # Nesnenin sınırlayıcı kutu (bounding box) koordinatlarını alıyoruz
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Takip kimliğini (track ID) alıyoruz
            track_id = box.id.int().item()

            # Sınırlayıcı kutuyu karenin üzerine çiziyoruz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Sınıf adı, güven skoru ve takip ID'sini kutunun üzerine yazıyoruz
            etiket = f"{sinif_adi} {guven_skoru:.2f} ID:{track_id}"
            cv2.putText(frame,
                        etiket,
                        (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # --- SAYIM BİLGİLERİNİ EKRANA YAZDIRMA ---

    # Sayım bilgilerini yazdırmak için başlangıç pozisyonu
    y_pozisyonu = 30

    # Her bir hedef sınıf için sayım bilgisini ekrana yazdırıyoruz
    for sinif, sayi in nesne_sayaci.items():
        sayac_metni = f"{sinif.capitalize()}: {sayi}"
        cv2.putText(
            frame,
            sayac_metni,
            (10, y_pozisyonu),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        y_pozisyonu += 30  # Bir sonraki metin için pozisyonu aşağı kaydır

    # İşlenmiş kareyi ekranda gösteriyoruz
    cv2.imshow("Gerçek Zamanlı Trafik Analizi", frame)

    # 'q' tuşuna basıldığında döngüden çıkılmasını sağlıyoruz
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- TEMİZLİK ---

# Video yakalama nesnesini serbest bırakıyoruz
cap.release()
# Tüm OpenCV pencerelerini kapatıyoruz
cv2.destroyAllWindows()

print("Uygulama başarıyla kapatıldı.")
