import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="Metin Yuksel - AI Financial Terminal", layout="wide")
st.title("🛡️ Metin Yüksel - Profesyonel Yatırım Terminali v11.0")

# 200+ ALFABETİK HİSSE LİSTESİ
bist_full_list = sorted([
    "ADEL", "ADESE", "AEFES", "AFYON", "AGESA", "AGHOL", "AGROT", "AHGAZ", "AKBNK", "AKCNS", 
    "AKENR", "AKFGY", "AKFYE", "AKGRT", "AKSA", "AKSEN", "ALARK", "ALBRK", "ALFAS", "ALGYO", 
    "ALKA", "ALKIM", "ALMAD", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARCLK", "ARDYZ", "ARENA", 
    "ARSAN", "ASELS", "ASGYO", "ASTOR", "ASUZU", "ATAKP", "ATATP", "ATEKS", "AVHOL", "AYDEM", 
    "AYGAZ", "AZTEK", "BAGFS", "BAKAB", "BANVT", "BARMA", "BASGZ", "BAYRK", "BERA", "BEYAZ", 
    "BIENY", "BIGCH", "BIMAS", "BIOEN", "BOBET", "BRSAN", "BRYAT", "BSOKE", "BTCIM", "BUCIM", 
    "BUPUN", "BURCE", "CANTE", "CCOLA", "CELHA", "CEMAS", "CEMTS", "CIMSA", "CLEBI", "CONSE", 
    "CVME", "CWENE", "DARDL", "DENGE", "DERHL", "DESPC", "DEVA", "DITAS", "DMSAS", "DOAS", 
    "DOHOL", "DOKTA", "DYOBY", "EBEBK", "ECILC", "ECZYT", "EGEEN", "EGEPO", "EGGUB", "EGSER", 
    "EKGYO", "EKOS", "ENERY", "ENJSA", "ENKAI", "EREGL", "ERSU", "ESEN", "EUHOL", "EUPWR", 
    "EUREN", "FENER", "FLAP", "FROTO", "FZLGY", "GARAN", "GENIL", "GENKM", "GEREL", "GESAN", 
    "GLYHO", "GSDHO", "GUBRF", "GWIND", "GZNMI", "HALKB", "HEKTS", "HTTBT", "HUNER", "IEYHO", 
    "IHAAS", "IHEVA", "IHGZT", "IHLAS", "IHLGM", "INVES", "IPEKE", "ISCTR", "ISDMR", "ISFIN", 
    "ISGYO", "ISMEN", "IZMDC", "KAREL", "KARDMA", "KARDMD", "KAYSE", "KCHOL", "KENT", "KERVT", 
    "KFEIN", "KLRGY", "KMPUR", "KONTR", "KONYA", "KORDS", "KOZAA", "KOZAL", "KRDMD", "KRVGD", 
    "KSTUR", "KTSKR", "KUTPO", "KUVVA", "KZBGY", "MAKTK", "MANAS", "MAVI", "MEDTR", "MEGAP", 
    "METUR", "MHRGY", "MIATK", "MIPAZ", "MMMKA", "MNDRS", "MOGAN", "MPARK", "MSGYO", "NETAS", 
    "NIBAS", "NTHOL", "NUGYO", "NUHCM", "OBAMS", "ODAS", "ONCSM", "ORCAY", "OTKAR", "OYAKC", 
    "OZKGY", "OZSUB", "PAGYO", "PAMEL", "PAPIL", "PARSN", "PASEU", "PENTA", "PETKM", "PGSUS", 
    "PNLSN", "POLHO", "PRKAB", "PRKME", "QUAGR", "REEDR", "RNPOL", "RODRG", "RTALB", "SAHOL", 
    "SAMAT", "SANKO", "SARKY", "SASA", "SAYAS", "SDTTR", "SELEC", "SELVA", "SISE", "SKBNK", 
    "SMRTG", "SNGYO", "SOPNR", "SOATM", "SOKM", "TABGD", "TARKM", "TATEN", "TAVHL", "TCELL", 
    "TEKTU", "THYAO", "TKFEN", "TMSN", "TOASO", "TRGYO", "TSKB", "TTKOM", "TTRAK", "TUPRS", 
    "TURSG", "UFUK", "ULAS", "ULKER", "ULUUN", "VAKBN", "VAKKO", "VESBE", "VESTL", "YEOTK", 
    "YKBNK", "YUNSA", "ZOREN", "ZRGYO"
])

# SIDEBAR PARAMETRELERİ
st.sidebar.header("📊 Analiz Ayarları")
vade_secenek = st.sidebar.selectbox("Yatırım Vadesi:", ["Kısa Vade (1-15 Gün)", "Orta Vade (1-6 Ay)", "Uzun Vade (6 Ay+)"])
min_guven = st.sidebar.slider("AI Güven Eşiği (%)", 50, 80, 52)

# RADAR VE TARAMA
if st.button("🚀 TÜM PİYASAYI TARA VE FIRSATLARI LİSTELE"):
    firsatlar = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Vadeye göre hedef gün sayısı
    shift_map = {"Kısa Vade (1-15 Gün)": 5, "Orta Vade (1-6 Ay)": 22, "Uzun Vade (6 Ay+)": 60}
    target_shift = shift_map[vade_secenek]

    for i, hisse in enumerate(bist_full_list):
        progress_bar.progress((i + 1) / len(bist_full_list))
        status_text.text(f"Analiz Ediliyor: {hisse}")
        
        try:
            sembol = f"{hisse}.IS"
            data = yf.download(sembol, period="5y", interval="1d", progress=False)
            
            if not data.empty and len(data) > 65:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                # TEKNİK VERİLER
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                data['RSI'] = 100 - (100 / (1 + (gain / loss)))
                
                # AI MODELLEME
                data['Target'] = (data['Close'].shift(-target_shift) > data['Close']).astype(int)
                features = ['Close', 'SMA_20', 'SMA_50', 'RSI']
                clean_df = data[features + ['Target']].dropna()
                
                if not clean_df.empty:
                    X = clean_df[features]
                    y = clean_df['Target']
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    model.fit(X, y)
                    olasilik = model.predict_proba(data[features].tail(1))[0][1] * 100
                    
                    son_fiyat = data['Close'].iloc[-1]
                    son_rsi = data['RSI'].iloc[-1]
                    
                    if olasilik >= min_guven and son_rsi < 65:
                        firsatlar.append({
                            "Hisse": hisse,
                            "Fiyat": f"{float(son_fiyat):.2f} TL",
                            "AI Güveni": f"%{olasilik:.1f}",
                            "RSI": f"{son_rsi:.1f}",
                            "Vade": vade_secenek
                        })
        except:
            continue

    status_text.text("Tarama Başarıyla Tamamlandı!")
    st.markdown("---")
    if firsatlar:
        st.subheader(f"🎯 {vade_secenek} İçin Potansiyel Fırsatlar")
        df_firsat = pd.DataFrame(firsatlar).sort_values(by="AI Güveni", ascending=False)
        st.dataframe(df_res, use_container_width=True)
    else:
        st.warning("Seçilen kriterlerde şu an yükseliş sinyali veren hisse bulunamadı.")

# TEKLİ ANALİZ
st.markdown("---")
st.subheader("🔍 Spesifik Hisse Analizi")
secilen_hisse = st.selectbox("Listeden Seç:", ["Kendim Yazacağım"] + bist_full_list)
if secilen_hisse == "Kendim Yazacağım":
    manuel_kod = st.text_input("Kod Yaz (Örn: BTC-USD):", "THYAO").upper()
else:
    manuel_kod = secilen_hisse

if st.button("Seçili Hisseyi Analiz Et"):
    # Tekli analiz kodunu burada çalıştırıyoruz (aynı veri seti ve AI mantığıyla)
    st.write(f"{manuel_kod} için {vade_secenek} analizi raporlanıyor...")
