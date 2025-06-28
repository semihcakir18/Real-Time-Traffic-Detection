# Gerçek Zamanlı Trafik Analizi Sistemi

Bu proje, bir video akışı üzerinden gerçek zamanlı olarak araçları tespit etmek, her birine benzersiz bir kimlik atayarak takip etmek ve belirlenen bir sanal çizgiyi geçen araçları saymak için geliştirilmiştir.

## Demo

![Ekran görüntüsü 2025-06-28 145330](https://github.com/user-attachments/assets/d9bbabd6-fcb0-4cc3-be8c-60b08de27709)


## Özellikler

- **Gerçek Zamanlı Nesne Tespiti:** Videodaki arabaları, kamyonları, otobüsleri ve motosikletleri anlık olarak tespit eder.
- **Nesne Takibi (Object Tracking):** Tespit edilen her araca benzersiz bir ID atar ve araç ekranda kaldığı sürece bu ID ile takip eder.
- **Sanal Çizgi Sayacı:** Ekranda tanımlanan sanal bir çizgiyi geçen araçların sayısını tutar.
- **Yol Takibi Görselleştirmesi:** Her aracın izlediği yolu görsel olarak çizer.

## Kullanılan Teknolojiler

- **Python 3.9+**
- **OpenCV:** Görüntü ve video işleme için.
- **PyTorch:** Derin öğrenme altyapısı.
- **YOLOv8:** Hızlı ve isabetli nesne tespiti ve takibi için.

## Kurulum

1.  Bu repoyu klonlayın:
    ```bash
    git clone https://github.com/semihcakir18/Real-Time-Traffic-Detection
    cd proje-adiniz
    ```

2.  Bir sanal ortam (virtual environment) oluşturup aktif hale getirin:
    ```bash
    python -m venv venv
    ```
    ### Windows için:
     ```bash
    venv\Scripts\activate
    ```
    
    ### macOS/Linux için:
     ```bash
    source venv/bin/activate
    ```



3.  Gerekli kütüphaneleri `requirements.txt` dosyasından yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

Uygulamayı çalıştırmak için aşağıdaki komutları kullanabilirsiniz.

- **Bir video dosyası ile çalıştırmak için:**
  (Proje içindeki `videos` klasörüne kendi videonuzu ekleyip dosya yolunu güncelleyebilirsiniz.)
  ```bash
  python main.py
