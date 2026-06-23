# Supply Chain Risk Intelligence: AI-Driven Decision Support System

## 1. Latar Belakang dan Masalah Bisnis

Dalam manajemen logistik, merespons gangguan operasional—seperti konflik regional atau bencana alam—secara cepat adalah hal krusial untuk mencegah kerugian. Namun, informasi mengenai gangguan ini sering kali muncul dalam bentuk teks berita yang tidak terstruktur. Manajer operasional kehilangan waktu berharga jika harus membaca dan menafsirkan teks ini secara manual, sementara model *machine learning* tradisional tidak dirancang untuk mencerna data non-numerik secara langsung.

Proyek ini dibangun untuk menjembatani celah tersebut. Sistem ini merupakan purwarupa *Decision Support System* (DSS) yang mengotomatisasi konversi informasi tak terstruktur menjadi keputusan bisnis. 

Pendekatan penyelesaian masalah dilakukan melalui dua tahap komputasi utama:
1. **Ekstraksi Konteks (DeepSeek LLM):** Membaca teks berita logistik secara otomatis untuk mengekstrak variabel operasional penting (lokasi, estimasi durasi penundaan, dan tingkat keparahan gangguan).
2. **Estimasi Risiko (XGBoost Classifier):** Menggunakan data historis untuk menghitung probabilitas kegagalan rute logistik berdasarkan variabel yang telah diekstrak pada tahap pertama.

Hasil akhirnya bukan sekadar prediksi statistik, melainkan sebuah dasbor yang secara langsung mengalkulasi rasio eksposur finansial. Sistem memberikan rekomendasi mitigasi yang jelas kepada pengguna (seperti peringatan untuk mengalihkan rute) sebelum gangguan tersebut berdampak pada aset kargo.
