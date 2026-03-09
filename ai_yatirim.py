import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="Metin Yuksel - Borsa Terminali", layout="wide")
st.title("🛡️ Metin Yüksel - Profesyonel Yatırım Terminali v14.0")

# 200+ HİSSE LİSTESİ
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

st.sidebar.header("🎯 Strateji Merkezi")
vade_secenek = st.sidebar.selectbox("Tahmin Vadesi:", ["Kısa Vade (1-15 Gün)", "Orta Vade (1-6 Ay)", "Uzun Vade (6 Ay+)"])
min_guven = st.sidebar.slider("AI Filtresi (Min Güven %)", 50, 80, 52)

# RADAR TARAMA
if st.button("🚀 TÜM PİYASAYI TARA VE FIRSATLARI LİSTELE"):
    firsatlar = []
    pb = st.progress(0)
    st_msg = st.empty()
    shift_map = {"Kısa Vade (1-15 Gün)": 5, "Orta Vade (1-6 Ay)": 22, "Uzun Vade (6 Ay+)": 60}
    t_shift = shift_map[vade_secenek]

    for i, h in enumerate(bist_full_list):
        pb.progress((i + 1) / len(bist_full_list))
        st_msg.text(f"Analiz Ediliyor: {h}")
        try:
            d = yf.download(f"{h}.IS", period="2y", interval="1d", progress=False)
            if not d.empty and len(d) > 60:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                d['SMA_20'] = d['Close'].rolling(window=20).mean()
                d['SMA_50'] = d['Close'].rolling(window=50).mean()
                delta = d['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                d['RSI'] = 100 - (100 / (1 + (gain / loss)))
                d['Target'] = (d['Close'].shift(-t_shift) > d['Close']).astype(int)
                feat = ['Close', 'SMA_20', 'SMA_50', 'RSI']
                clean = d[feat + ['Target']].dropna()
                if not clean.empty:
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    model.fit(clean[feat], clean['Target'])
                    prob = model.predict_proba(d[feat].tail(1))[0][1] * 100
                    if prob >= min_guven and d['RSI'].iloc[-1] < 65:
                        firsatlar.append({"Hisse": h, "AI Güveni": f"%{prob:.1f}", "RSI": f"{d['RSI'].iloc[-1]:.1f}"})
        except: continue
    
    st_msg.text("Tarama Bitti!")
    if firsatlar:
        df_f = pd.DataFrame(firsatlar).sort_values(by="AI Güveni", ascending=False)
        st.subheader(f"✅ {vade_secenek} Fırsat Listesi")
        st.table(df_f)
    else:
        st.warning("Kriterlere uygun hisse bulunamadı.")

st.markdown("---")

# DERİN RÖNTGEN VE NET ANALİZ
st.subheader("🔍 Seçili Hisse İçin Derin Röntgen")
detay_hisse = st.selectbox("Hisse Seçin:", bist_full_list)

if st.button("📊 Röntgeni Çek"):
    with st.spinner('Derin analiz yapılıyor...'):
        d = yf.download(f"{detay_hisse}.IS", period="2y", interval="1d")
        if not d.empty:
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            
            # TEKNİK VERİLERİ YENİDEN HESAPLA (Hata almamak için)
            d['SMA_20'] = d['Close'].rolling(20).mean()
            d['SMA_50'] = d['Close'].rolling(50).mean()
            d['SMA_200'] = d['Close'].rolling(200).mean()
            delta = d['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            d['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI TAHMİNİNİ RÖNTGEN İÇİN TEKRAR ÇALIŞTIR
            shift_map = {"Kısa Vade (1-15 Gün)": 5, "Orta Vade (1-6 Ay)": 22, "Uzun Vade (6 Ay+)": 60}
            t_shift = shift_map[vade_secenek]
            d['Target'] = (d['Close'].shift(-t_shift) > d['Close']).astype(int)
            feat = ['Close', 'SMA_20', 'SMA_50', 'RSI']
            clean = d[feat + ['Target']].dropna()
            
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(clean[feat], clean['Target'])
            olasilik = model.predict_proba(d[feat].tail(1))[0][1] * 100

            fiyat = d['Close'].iloc[-1]
            rsi = d['RSI'].iloc[-1]
            sma20 = d['SMA_20'].iloc[-1]
            sma200 = d['SMA_200'].iloc[-1]

            st.markdown(f"### {detay_hisse} Stratejik Analiz Raporu")
            st.line_chart(d[['Close', 'SMA_20', 'SMA_50']].tail(120))

            # HEDEF FİYAT (Basit mühendislik tahmini: AI güveni ve volatilite bazlı)
            hedef_fiyat = fiyat * (1 + (olasilik/500)) if olasilik > 50 else fiyat

            c1, c2 = st.columns(2)
            with c1:
                st.info("📌 **Teknik Durum**")
                st.write(f"**Anlık Fiyat:** {float(fiyat):.2f} TL")
                st.write(f"**AI Beklentisi (Hedef):** {float(hedef_fiyat):.2f} TL")
                st.write(f"**RSI Gücü:** {rsi:.2f}")
            
            with c2:
                st.info("🤖 **AI Strateji Notu**")
                if rsi < 28 and olasilik > 55:
                    st.success("🔥 **BÜYÜK FIRSAT:** Hisse bedava kalmış. Alım için harika bir yer!")
                elif olasilik > 55 and fiyat > sma20:
                    st.success("🚀 **ALINABİLİR:** Yükseliş trendi başlamış. AI artış bekliyor.")
                elif rsi < 35:
                    st.warning("💎 **DİPTE AMA BEKLE:** Çok ucuz ama henüz dönüş yok. Kademeli alım yapılabilir.")
                elif olasilik < 45:
                    st.error("❌ **ALINMAZ / UZAK DUR:** AI düşüş bekliyor. Macera aramaya gerek yok.")
                else:
                    st.write("🔎 **BELİRSİZ:** Net bir 'Al' sinyali yok. Nakitte kalmak daha mantıklı.")

            st.caption(f"Yasal Uyarı: AI tahmini %{olasilik:.1f} güven vermektedir. Yatırım tavsiyesi değildir.")
