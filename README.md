# Supply Chain Risk Intelligence: End-to-End AI-Driven Decision Support System

## 1. Latar Belakang dan Konteks Bisnis
Dalam manajemen logistik global, informasi mengenai disrupsi operasional—seperti konflik geopolitik, pemogokan kerja, hingga bencana alam—sering kali tersebar dalam bentuk data tekstual tidak terstruktur seperti berita atau laporan intelijen. Tantangan utamanya adalah lambatnya proses konversi informasi tersebut menjadi keputusan taktis yang terukur secara finansial.

Proyek ini membangun sebuah purwarupa **Decision Support System (DSS)** hibrida. Sistem ini mengintegrasikan kapabilitas eksplorasi bahasa alami dari Large Language Model (LLM) untuk mengekstrak parameter operasional, kemudian menyelaraskannya ke dalam mesin inferensi statistik guna memproyeksikan probabilitas kegagalan rute serta eksposur kerugian finansial secara real-time.

```text
[Alur Kerja Komputasi]
Teks Berita (Unstructured) -> DeepSeek API (Resolusi NER) -> Parameter Terstruktur (JSON)
                                                                       |
                                                                       v
Keputusan Taktis (Streamlit) <-- Kalkulator Finansial <-- Mesin Inferensi XGBoost (.pkl)
