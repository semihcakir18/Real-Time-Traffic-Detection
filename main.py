# Gerekli kütüphaneleri içe aktarıyoruz
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# --- TEMEL AYARLAR ---

# Önceden eğitilmiş YOLOv8 modelini yüklüyoruz.
model = YOLO("yolov8n.pt")

# İşlem yapılacak video kaynağını belirtiyoruz.
video_kaynagi = "videos/traffic.mp4"

# Video yakalama nesnesini oluşturuyoruz
cap = cv2.VideoCapture(video_kaynagi)

# Sadece belirli nesneleri tespit etmek için bir filtre listesi oluşturuyoruz.
# Bu projede sadece araçları sayacağımız için listeyi basitleştirebiliriz.
hedef_siniflar = ["car", "truck", "bus", "motorcycle"]

# --- BÖLGE ANALİZİ İÇİN AYARLAR ---

# Sanal çizgi koordinatları (videonuzun boyutlarına göre ayarlayın)
# Bu örnekte, videonun ortasına yakın yatay bir çizgi tanımlıyoruz.
LINE_START = (100, 400)
LINE_END = (1180, 400)

# Nesne takip geçmişini saklamak için bir sözlük
# defaultdict, bir anahtar ilk kez erişildiğinde varsayılan bir değer oluşturur.
track_history = defaultdict(lambda: [])

# Çizgiyi geçen nesneleri saymak için
crossing_counter = 0
crossed_ids = set()  # Zaten sayılmış nesne ID'lerini saklamak için

# --- ANA DÖNGÜ ---

while True:
    # Videodan bir kare (frame) okuyoruz
    success, frame = cap.read()

    # Eğer video bittiyse veya kare okunamadıysa döngüden çıkıyoruz
    if not success:
        print("Video bitti veya kare okunamadı.")
        break

    # Okunan kareyi YOLOv8 modeline vererek nesne takibi yapıyoruz
    # 'persist=True' takibin kareler arasında devamlılığını sağlar.
    results = model.track(frame, persist=True, stream=True, verbose=False)

    # Tespit edilen sonuçlar üzerinde döngüye giriyoruz
    for r in results:
        # Tespit edilen nesnelerin kutularını alıyoruz
        boxes = r.boxes.xywh.cpu()

        # Takip edilen nesne yoksa track_ids None olabilir, bu durumu kontrol edelim.
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()

            # Her bir nesne için işlem yapıyoruz
            for box, track_id in zip(boxes, track_ids):
                # Sınıf adını alıyoruz (bu örnekte sınıfı filtrelemiyoruz ama isterseniz ekleyebilirsiniz)
                # cls_id = int(box.cls[0])
                # sinif_adi = model.names[cls_id]
                # if sinif_adi not in hedef_siniflar:
                #    continue

                # Nesnenin sınırlayıcı kutu (bounding box) koordinatlarını alıyoruz
                x, y, w, h = box
                center_x, center_y = int(x), int(y)  # Merkez noktası

                # Takip geçmişini güncelle
                track = track_history[track_id]
                track.append((float(center_x), float(center_y)))  # x, y merkez noktalarını ekle
                if len(track) > 15:  # Geçmişi çok uzatmamak için eski noktaları sil
                    track.pop(0)

                # Nesnenin takip yolunu çizdiriyoruz (görselleştirme için)
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                # --- ÇİZGİ GEÇİŞ KONTROLÜ ---
                # Eğer nesnenin en az iki pozisyonu varsa (önceki ve şimdiki)
                if len(track) > 1:
                    prev_point_y = track[-2][1]  # Önceki y koordinatı
                    curr_point_y = track[-1][1]  # Şimdiki y koordinatı
                    line_y = LINE_START[1]

                    # Nesne yukarıdan aşağıya mı geçti?
                    if prev_point_y < line_y and curr_point_y >= line_y and track_id not in crossed_ids:
                        # Sadece çizginin x ekseni aralığındaki geçişleri say
                        if LINE_START[0] < center_x < LINE_END[0]:
                            crossed_ids.add(track_id)
                            crossing_counter += 1

                    # Nesne aşağıdan yukarıya mı geçti? (İsteğe bağlı, yönlü sayım için)
                    elif prev_point_y > line_y and curr_point_y <= line_y and track_id not in crossed_ids:
                        if LINE_START[0] < center_x < LINE_END[0]:
                            crossed_ids.add(track_id)
                            crossing_counter += 1

                # Nesnenin etrafına bir daire çizerek merkezini gösteriyoruz
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                # Takip ID'sini yazdırıyoruz
                cv2.putText(frame, f"ID:{track_id}", (center_x, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # --- GÖRSELLEŞTİRME ---

    # Sayım çizgisini ekrana çizdiriyoruz
    cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 3)

    # Geçen araç sayısını ekrana yazdırıyoruz
    cv2.putText(frame, f"Gecen Arac Sayisi: {crossing_counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

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
