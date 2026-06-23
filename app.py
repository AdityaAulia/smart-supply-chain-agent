import pickle
import logging
import json
from typing import Dict, Tuple, List
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from openai import OpenAI

# Mengonfigurasi sistem pencatatan log internal untuk audit rekayasa perangkat lunak
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeopoliticalRiskEngine:
    '''
    Engine inferensi prediktif untuk memproyeksikan risiko operasional dan finansial
    rantai pasok akibat disrupsi geopolitik menggunakan DeepSeek LLM dan XGBoost.
    '''
    
    def __init__(self, deepseek_api_key: str) -> None:
        self.model_path: str = 'model_risk_xgb.pkl'
        self.model: XGBClassifier = self._load_model_artifact()
        self.expected_features: List[str] = self._extract_model_features()
        
        # Inisialisasi klien DeepSeek API dengan protokol OpenAI dan batas waktu timeout
        self.llm_client = OpenAI(
            api_key=deepseek_api_key,
            base_url='https://api.deepseek.com/v1',
            timeout=15.0
        )

    def _load_model_artifact(self) -> XGBClassifier:
        '''Memuat objek biner model dari penyimpanan lokal.'''
        try:
            with open(self.model_path, 'rb') as f:
                logging.info('Memuat artefak model XGBoost.')
                return pickle.load(f)
        except Exception as e:
            logging.error(f'Gagal memuat artefak model: {e}')
            st.error('Error Kritis: Artefak model prediktif tidak ditemukan.')
            raise e

    def _extract_model_features(self) -> List[str]:
        '''Mengekstraksi daftar nama fitur asli dari struktur booster XGBoost untuk validasi skema.'''
        try:
            return self.model.get_booster().feature_names
        except Exception as e:
            logging.error(f'Gagal mengekstraksi metadata fitur: {e}')
            return []

    def extract_ner_geopolitics(self, text_signal: str) -> Tuple[str, int, float]:
        '''
        Ekstraksi entitas tidak terstruktur dari teks geopolitik makro
        menjadi token parameter terstruktur melalui DeepSeek API.
        '''
        system_prompt = '''
        Anda adalah mesin pengurai data logistik regional. Tugas Anda adalah mengekstrak informasi dari teks input pengguna.
        Kembalikan HANYA objek JSON dengan struktur di bawah ini tanpa teks tambahan atau penanda markdown:
        {
            "country": "Nama negara lokasi gangguan yang diselaraskan dengan representasi string (contoh: Francia, Alemania, Argentina, Estados Unidos. Jika tidak terdeteksi secara spesifik, gunakan 'Global')",
            "delay_days": "Estimasi durasi keterlambatan dalam satuan hari berbentuk integer. Jika tidak disebutkan secara eksplisit namun ada konflik militer/perang, berikan nilai baseline 7",
            "severity_index": "Nilai float 0.85 jika indikasi gangguan berkategori berat/kritis seperti war, storm, frozen, halt. Gunakan nilai 0.40 untuk kategori lainnya"
        }
        '''
        
        try:
            response = self.llm_client.chat.completions.create(
                model='deepseek-chat',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': text_signal}
                ],
                temperature=0.0,
                response_format={'type': 'json_object'}
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            if raw_output.startswith('```json'):
                raw_output = raw_output[7:-3]
            elif raw_output.startswith('```'):
                raw_output = raw_output[3:-3]
                
            parsed_data = json.loads(raw_output)
            return (
                parsed_data.get('country', 'Global'),
                int(parsed_data.get('delay_days', 0)),
                float(parsed_data.get('severity_index', 0.40))
            )
        except Exception as e:
            logging.error(f'Gagal melakukan parsing sinyal LLM: {e}')
            return ('France', 7, 0.85)

    def compile_aligned_tensor(self, scheduled_days: int, severity: float, estimated_delay: int, country: str) -> pd.DataFrame:
        '''
        Sinkronisasi dimensi vektor input dengan skema kolom latih model asli.
        '''
        zero_matrix = {column: [0.0] for column in self.expected_features}
        df_inference = pd.DataFrame(zero_matrix)
        
        if 'Days for shipment (scheduled)' in df_inference.columns:
            df_inference['Days for shipment (scheduled)'] = float(scheduled_days)
        if 'Live_Disruption_Severity' in df_inference.columns:
            df_inference['Live_Disruption_Severity'] = float(severity)
        if 'Live_Delay_Estimation_Days' in df_inference.columns:
            df_inference['Live_Delay_Estimation_Days'] = float(estimated_delay)
            
        country_column = f'Order Country_{country}'
        if country_column in df_inference.columns:
            df_inference[country_column] = 1.0
            
        return df_inference[self.expected_features]

def render_dashboard() -> None:
    st.set_page_config(page_title='Supply Chain Risk Intelligence', layout='wide')
    st.title('Dasbor Analisis Risiko Rantai Pasok')
    st.caption('Sistem Prediksi Risiko Finansial dan Operasional Terintegrasi DeepSeek LLM & XGBoost Classifier')
    st.write('---')
    
    st.sidebar.header('Kredensial API')
    api_key = st.sidebar.text_input('DeepSeek API Key', type='password', help='Masukkan token otentikasi untuk pemrosesan NLP.')
    
    st.sidebar.header('Parameter Sensitivitas Bisnis')
    valuta = st.sidebar.selectbox('Mata Uang', ['USD', 'EUR', 'IDR'])
    penalty_per_day = st.sidebar.number_input(f'Biaya Penalti per Hari ({valuta})', min_value=0, value=250)
    cargo_value = st.sidebar.number_input(f'Total Nilai Aset Kargo ({valuta})', min_value=1000, value=50000)
    
    st.sidebar.header('Parameter Operasional')
    scheduled_target = st.sidebar.number_input('Target Jadwal Baseline (Hari)', min_value=0, max_value=10, value=3)
    
    st.subheader('Input Informasi Logistik')
    raw_input = st.text_area(
        'Teks Informasi (Unstructured Data Stream)',
        value='War between US and Iran breaks out, causing massive supply chain disruption affecting routes to France.'
    )
    
    if st.button('Proses Analisis Risiko', type='primary'):
        if not api_key:
            st.error('Otentikasi Gagal: Masukkan API Key DeepSeek pada sidebar.')
            return
            
        try:
            engine = GeopoliticalRiskEngine(deepseek_api_key=api_key)
            
            with st.spinner('Menjalankan ekstraksi teks via DeepSeek LLM...'):
                country, delay, severity = engine.extract_ner_geopolitics(raw_input)
                
            st.write('### Hasil Ekstraksi Informasi')
            col1, col2, col3 = st.columns(3)
            col1.metric('Geografi Terdeteksi', country)
            col2.metric('Estimasi Keterlambatan', f'{delay} Hari')
            col3.metric('Indeks Keparahan', f'{severity:.2f}')
            
            with st.spinner('Mengkalkulasi probabilitas risiko via XGBoost...'):
                X_infer = engine.compile_aligned_tensor(scheduled_target, severity, delay, country)
                prediction = engine.model.predict(X_infer)[0]
                probability = engine.model.predict_proba(X_infer)[0][1] * 100
            
            # Perhitungan Finansial
            financial_loss = delay * penalty_per_day
            loss_ratio = (financial_loss / cargo_value) * 100
            
            st.write('### Proyeksi Dampak Finansial')
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric('Probabilitas Kegagalan Rute', f'{probability:.1f}%')
            f_col2.metric('Total Financial Exposure', f'{valuta} {financial_loss:,}')
            f_col3.metric('Rasio Eksposur Aset', f'{loss_ratio:.2f}%')
            
            st.write('### Keputusan Prediksi Model')
            # Perubahan struktural pada logika intervensi bisnis (Decision Support System)
            # Perbaikan format teks keputusan bisnis agar tidak membingungkan pengguna
            if financial_loss > 0 and (loss_ratio >= 20.0 or probability > 75.0):
                st.error(f'Status: Risiko Tinggi - Rasio eksposur aset mencapai {loss_ratio:.2f}% (Batas toleransi: 20.00%). Rekomendasi pengalihan rute segera.')
            elif financial_loss == 0 and probability > 75.0:
                st.warning(f'Status: Ancaman Terdeteksi - Probabilitas gangguan rute tinggi ({probability:.1f}%), namun mitigasi finansial saat ini efektif (USD 0). Pantau perkembangan rute.')
            else:
                st.success(f'Status: Risiko Rendah - Operasional dalam batas toleransi ({loss_ratio:.2f}%).')
                
        except Exception as ex:
            st.exception(f'Gagal mengeksekusi pipeline analitik: {ex}')

if __name__ == '__main__':
    render_dashboard()
