# Traffic Monitoring System Semarang

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Framework-Flask-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/AI-YOLOv8-yellow.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Frontend-Bootstrap-purple.svg" alt="Bootstrap">
  <img src="https://img.shields.io/badge/License-MIT-red.svg" alt="License">
</div>

<p align="center">
  <img src="https://i.imgur.com/zS2Z6KE.png" alt="Traffic Monitoring System Preview">
</p>

## 📋 Deskripsi Singkat Project

**Monitoring Lalu Lintas Semarang** adalah sistem monitoring lalu lintas real-time yang menggabungkan AI computer vision dengan analisis jaringan jalan untuk membantu pengguna menavigasi kota Semarang dengan efisien. Sistem ini menggunakan model YOLOv8 untuk mendeteksi dan menghitung kendaraan dari delapan lokasi CCTV strategis di Semarang, kemudian mengklasifikasikan kondisi lalu lintas sebagai "Lancar", "Padat", atau "Macet" berdasarkan kepadatan kendaraan.

### Fitur Utama:
- 🚗 **Deteksi Kendaraan Real-time** menggunakan YOLOv8 pada feed CCTV
- 🚦 **Klasifikasi Kondisi Lalu Lintas** berdasarkan jumlah kendaraan terdeteksi
- 🗺️ **Rekomendasi Rute Optimal** dengan algoritma Dijkstra
- 📊 **Visualisasi Peta Jaringan** dengan kondisi lalu lintas terkini
- 📱 **Antarmuka Responsif** yang bekerja di semua ukuran perangkat

## 🗂️ Struktur Project

```
.
├── app.py                 # Aplikasi Flask utama dengan logika backend
├── README.md              # Dokumentasi project (file ini)
├── requirements.txt       # Daftar dependensi Python
├── templates
│   └── index.html         # Halaman antarmuka pengguna
└── yolov8l.pt             # Model YOLOv8 pre-trained
```

### Komponen Utama:
- **app.py**: Berisi implementasi server Flask, logika deteksi kendaraan menggunakan YOLOv8, algoritma perhitungan rute, dan API endpoint.
- **index.html**: Antarmuka berbasis web dengan tampilan responsif yang menampilkan feed CCTV, status lalu lintas, dan rekomendasi rute.
- **requirements.txt**: Daftar library Python yang diperlukan untuk menjalankan project.
- **yolov8l.pt**: Model YOLO pre-trained yang digunakan untuk deteksi kendaraan.

## 🚀 Cara Menjalankan Project di Windows

### Prasyarat
- Python 3.8 atau lebih baru
- Git (opsional)
- Koneksi internet untuk download model YOLOv8 dan akses feed CCTV

### Langkah Instalasi

1. **Clone atau download repository ini**

   ```bash
   git clone https://github.com/notsuperganang/Monitoring-Lalu-Lintas-Semarang.git
   cd traffic-monitoring-semarang
   ```

   Atau download dan ekstrak ZIP dari repository.

2. **Buat dan aktifkan virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependensi**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download model YOLOv8** (jika tidak disertakan dalam repository)

   ```bash
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8l.pt')"
   ```

5. **Jalankan aplikasi**

   ```bash
   python app.py
   ```

6. **Akses aplikasi**
   
   Buka browser dan kunjungi: `http://127.0.0.1:5000`

## 🌐 Tunneling dengan Ngrok di Windows

Tunneling memungkinkan aplikasi lokal Anda diakses melalui internet. Berikut cara melakukannya dengan ngrok:

### 1. Buat Akun Ngrok

1. Kunjungi [ngrok.com](https://ngrok.com/) dan daftar untuk akun gratis
2. Verifikasi email Anda
3. Login ke dashboard ngrok

### 2. Instalasi Ngrok di Windows

1. **Download Ngrok**
   - Kunjungi [ngrok.com/download](https://ngrok.com/download)
   - Download versi Windows (.zip)

2. **Ekstrak file zip** ke lokasi yang mudah diakses, misalnya `C:\ngrok`

3. **Tambahkan Authtoken**
   - Salin authtoken dari dashboard ngrok Anda ([dashboard.ngrok.com/auth/your-authtoken](https://dashboard.ngrok.com/auth/your-authtoken))
   - Buka Command Prompt dan jalankan:
   ```bash
   cd C:\ngrok
   ngrok config add-authtoken YOUR_AUTHTOKEN
   ```

### 3. Jalankan Ngrok

1. **Pastikan aplikasi Flask Anda berjalan** (pada port 5000)

2. **Buka Command Prompt baru** dan jalankan:

   ```bash
   cd C:\ngrok
   ngrok http 5000
   ```

3. **Salin URL Ngrok**
   - Ngrok akan menampilkan URL forwarding, contoh: `https://a1b2c3d4.ngrok.io`
   - URL ini dapat digunakan untuk mengakses aplikasi dari internet

   ![Contoh Output Ngrok](https://i.imgur.com/N0rGIg0.png)

### 4. Akses Aplikasi Melalui Ngrok

1. Salin URL forwarding yang diberikan oleh ngrok
2. Bagikan URL ini dengan orang lain agar mereka dapat mengakses aplikasi Anda
3. URL ini aktif selama sesi ngrok berjalan di komputer Anda

### Catatan Penting:
- Akun gratis ngrok memiliki batasan sesi 2 jam (URL berubah setelah 2 jam)
- Untuk sesi yang lebih lama, pertimbangkan upgrade ke paket berbayar
- Pastikan komputer Anda tetap menyala dan terhubung ke internet selama URL ngrok ingin tetap aktif

## 📝 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<p align="center">
  Dibuat dengan ❤️ untuk Kota Semarang
</p>