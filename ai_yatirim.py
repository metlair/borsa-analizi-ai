import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="Metin Yüksel AI Terminal", layout="wide")
st.title("🚀 Metin Yüksel - Profesyonel Yatırım Terminali v3.0")

# 1. BIST LİSTESİ VE STRATEJİ SEÇİMİ
bist_hisseleri = ["THYAO", "ASELS", "FROTO", "PGSUS", "EREGL", "SASA", "KCHOL", "SISE", "AKBNK", "GARAN", "TUPRS", "ISCTR"]
secilen_hisse = st.selectbox("Analiz Edilecek Hisseyi Seçin:", bist_hisseleri)
vade = st.radio("Yatırım Stratejiniz Nedir?", ("Kısa Vade (1-15 Gün)", "Orta Vade (1-6 Ay)", "Uzun Vade (6 Ay - 1 Yıl)"))

if st.button("Kapsamlı Analizi Başlat"):
    with st.spinner('Piyasa verileri ve AI modelleri işleniyor...'):
        hisse_kodu = f"{secilen_hisse}.IS"
        
        # Vadeye göre veri derinliği
        if vade == "Kısa Vade (1-15 Gün)":
            periyot, shift_days = "1y", 5
        elif vade == "Orta Vade (1-6 Ay)":
            periyot, shift_days = "2y", 22
        else:
            periyot, shift_days = "5y", 66

        data = yf.download(hisse_kodu, period=periyot, interval="1d")
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # TEKNİK HESAPLAMALAR
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI MODELİ
            data['Target'] = (data['Close'].shift(-shift_days) > data['Close']).astype(int)
            data.dropna(inplace=True)
            X = data[['Close', 'SMA_20', 'RSI']]
            y = data['Target']
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            olasilik = model.predict_proba(X.tail(1))[0][1] * 100

            # GÖRSELLEŞTİRME
            st.subheader(f"📊 {secilen_hisse} Teknik Analiz Grafiği")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_200']].tail(250))

            st.markdown("---")
            st.subheader("🔍 AI Strateji Raporu")
            
            col1, col2 = st.columns(2)
            son_rsi = data['RSI'].iloc[-1]
            fiyat = data['Close'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]

            with col1:
                st.markdown("### 💼 Elinde Hisse Olanlar İçin")
                if olasilik > 65:
                    st.success(f"🚀 **TUT:** Trend güçlü (%{olasilik:.1f}). Pozisyonu korumak mantıklı görünüyor.")
                elif olasilik < 35:
                    st.error(f"⚠️ **DİKKAT:** Zayıflama var (%{100-olasilik:.1f}). Kar satışı veya stop-loss düşünülebilir.")
                else:
                    st.warning(f"⚖️ **İZLE:** Yatay seyir. Belirgin bir kırılım beklemek daha güvenli.")

            with col2:
                st.markdown("### 💰 Yeni Alım Yapacaklar İçin")
                if olasilik > 65 and son_rsi < 60:
                    st.success(f"✅ **ALIM FIRSATI:** AI artış bekliyor ve RSI henüz aşırı alımda değil. Kademeli alım uygun olabilir.")
                elif olasilik > 65 and son_rsi >= 60:
                    st.warning(f"⌛ **BEKLE:** AI olumlu ancak hisse kısa vadede çok primli. Küçük bir geri çekilme beklenebilir.")
                elif olasilik < 40:
                    st.error(f"❌ **UZAK DUR:** Düşüş trendi hakim. Alım için dip oluşumu beklenmeli.")
                else:
                    st.info(f"🔎 **GÖZLEM:** Net bir alım sinyali yok. Destek seviyelerine yaklaşması takip edilmeli.")

            # TEKNİK DETAY ÖZETİ
            st.markdown("---")
            detay1, detay2, detay3 = st.columns(3)
            detay1.metric("Anlık Fiyat", f"{fiyat:.2f} TL")
            detay2.metric("RSI (Güç Endeksi)", f"{son_rsi:.2f}")
            detay3.metric("AI Artış İhtimali", f"%{olasilik:.1f}")
            
        else:
            st.error("Veri çekilemedi. Lütfen internet bağlantınızı veya hisse kodunu kontrol edin.")
