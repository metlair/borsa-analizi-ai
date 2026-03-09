import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="Metin Yuksel - Yatirim Terminali", layout="wide")
st.title("📈 Metin Yüksel - Finansal Analiz ve Yatırım Terminali")

# 200+ POPÜLER HİSSE LİSTESİ (ALFABETİK)
bist_200 = sorted([
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
    "SMRTG", "SNGYO", "SOPNR", "SOATM", "SOKM", " TABGD", "TARKM", "TATEN", "TAVHL", "TCELL", 
    "TEKTU", "THYAO", "TKFEN", "TMSN", "TOASO", "TRGYO", "TSKB", "TTKOM", "TTRAK", "TUPRS", 
    "TURSG", "UFUK", "ULAS", "ULKER", "ULUUN", "VAKBN", "VAKKO", "VESBE", "VESTL", "YEOTK", 
    "YKBNK", "YUNSA", "ZOREN", "ZRGYO"
])

st.sidebar.header("Analiz Parametreleri")
secim = st.sidebar.selectbox("Hisse Seçin:", ["Kendim Yazacağım"] + bist_200)

if secim == "Kendim Yazacağım":
    hisse_adi = st.sidebar.text_input("Hisse Kodu Yazın (Örn: BTC-USD, ASUZU):", "THYAO").upper()
else:
    hisse_adi = secim

vade = st.sidebar.radio("Yatırım Vadesi:", ("Kısa Vade (1-15 Gün)", "Orta Vade (1-6 Ay)", "Uzun Vade (6 Ay+)"))

if st.sidebar.button("Analizi Gerçekleştir"):
    with st.spinner(f'{hisse_adi} verileri işleniyor...'):
        sembol = f"{hisse_adi}.IS" if len(hisse_adi) <= 6 and "-" not in hisse_adi else hisse_adi
        data = yf.download(sembol, period="5y", interval="1d")
        
        if not data.empty and len(data) > 60:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            # TEKNİK HESAPLAMALAR
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI MODELLEME
            shift_map = {"Kısa Vade (1-15 Gün)": 5, "Orta Vade (1-6 Ay)": 22, "Uzun Vade (6 Ay+)": 60}
            target_shift = shift_map[vade]
            data['Target'] = (data['Close'].shift(-target_shift) > data['Close']).astype(int)
            
            features = ['Close', 'SMA_20', 'SMA_50', 'RSI']
            clean_df = data[features + ['Target']].dropna()
            
            X = clean_df[features]
            y = clean_df['Target']
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X, y)
            olasilik = model.predict_proba(data[features].tail(1))[0][1] * 100

            # RİSK VE EMNİYET PARAMETRELERİ
            son_fiyat = data['Close'].iloc[-1]
            son_rsi = data['RSI'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]
            hacim_ort = data['Volume'].tail(10).mean()
            son_hacim = data['Volume'].iloc[-1]
            stop_loss = son_fiyat * 0.95 

            risk_skoru = 1
            if son_rsi > 70: risk_skoru += 3
            if son_hacim < hacim_ort: risk_skoru += 2
            if son_fiyat < sma20: risk_skoru += 2

            # GÖRSELLEŞTİRME
            st.subheader(f"📊 {hisse_adi} Teknik Analiz Paneli")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_50']].tail(150))

            m1, m2, m3 = st.columns(3)
            m1.metric("Anlık Fiyat", f"{float(son_fiyat):.2f} TL")
            m2.metric("Stop-Loss (Zarar Kes)", f"{float(stop_loss):.2f} TL")
            m3.metric("Analiz Risk Puanı", f"{risk_skoru}/10")

            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Mevcut Pozisyon Durumu")
                if olasilik > 55 and son_fiyat > sma20:
                    st.success(f"✅ **POZİSYONU KORU:** Trend ve AI desteği pozitif (%{olasilik:.1f}).")
                elif olasilik < 45:
                    st.error(f"⚠️ **RİSKLİ BÖLGE:** Düşüş ihtimali artıyor (%{100-olasilik:.1f}).")
                else:
                    st.warning("⚖️ **YATAY SEYİR:** Belirgin bir yön sinyali bulunmuyor.")

            with col2:
                st.markdown("### Yeni Giriş Stratejisi")
                if olasilik > 55 and son_hacim > hacim_ort and son_rsi < 60:
                    st.success("🎯 **ALUM UYGUN:** Hacim ve AI onayıyla giriş denenebilir.")
                elif son_rsi < 30:
                    st.info("🔎 **TAKİP ET:** Aşırı satım var ancak dönüş için hacim artışı beklenmeli.")
                else:
                    st.write("⌛ **BEKLE:** Alım için yeterli güvenli bölge oluşmadı.")
        else:
            st.error("Yeterli veri bulunamadı veya sembol hatalı.")
