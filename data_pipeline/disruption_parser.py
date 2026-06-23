import os
from openai import OpenAI
from pydantic import BaseModel, Field

class MetrikGangguan(BaseModel):
    # Skema output JSON yang wajib dipenuhi DeepSeek
    location_country: str = Field(description="Negara lokasi gangguan. Gunakan bahasa Spanyol (misal: France jadi Francia).")
    estimated_delay_days: int = Field(description="Estimasi penundaan dalam hari.")
    severity_score: float = Field(description="Skor keparahan dari 0.0 sampai 1.0.")

class AgenIntelijenRisiko:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("API Key 'DEEPSEEK_API_KEY' belum dipasang di environment.")
            
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.model_target = "deepseek-chat"

    def ekstrak_metrik_risiko(self, teks_berita_mentah: str) -> str:
        sistem_prompt = (
            "Anda adalah agen logistik. Tugas Anda hanya mengubah berita mentah menjadi "
            "format JSON yang valid sesuai skema. Jangan berikan teks pembuka atau penutup."
        )
        
        user_prompt = f"Ekstrak berita berikut ke JSON:\n\"{teks_berita_mentah}\""
        
        response = self.client.chat.completions.create(
            model=self.model_target,
            messages=[
                {"role": "system", "content": sistem_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

# Uji coba skrip
if __name__ == "__main__":
    berita_contoh = (
        "A severe storm hit major maritime hubs across France, freezing freight operations "
        "and triggering up to 4 days of container shipping delays at local terminals."
    )
    
    try:
        agent = AgenIntelijenRisiko()
        print("Menghubungkan ke DeepSeek...")
        hasil = agent.ekstrak_metrik_risiko(berita_contoh)
        print("\nHasil JSON:")
        print(hasil)
    except Exception as error:
        print(f"Gagal: {error}")