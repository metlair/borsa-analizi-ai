import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="AI Yatırım Terminali", layout="wide")
st.title("🚀 Metin Yüksel - Profesyonel Yatırım Terminali")

# 1. BIST 100 LİSTESİ
bist_hisseleri = ["THYAO", "ASELS", "FROTO", "PGSUS", "EREGL", "SASA", "KCHOL", "SISE", "AKBNK", "GARAN", "TUPRS", "ISCTR"]
secilen_hisse = st.selectbox("Analiz Edilecek Hisseyi Seçin:", bist_hisseleri)
vade = st.radio("Yatırım Stratejiniz Nedir?", ("Kısa Vade (1-15 Gün)", "Uzun Vade (6 Ay - 1 Yıl)"))

if st.button("Teknik Analizi Başlat"):
    with st.spinner('Derin analiz yapılıyor...'):
        hisse_kodu = f"{secilen_hisse}.IS"
        periyot = "1y" if vade == "Kısa Vade (1-15 Gün)" else "5y"
        data = yf.download(hisse_kodu, period=periyot, interval="1d")
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # 2. TEKNİK HESAPLAMALAR
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI MODELİ
            shift_days = 5 if vade == "Kısa Vade (1-15 Gün)" else 60
            data['Target'] = (data['Close'].shift(-shift_days) > data['Close']).astype(int)
            data.dropna(inplace=True)
            
            X = data[['Close', 'SMA_20', 'RSI']]
            y = data['Target']
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            
            olasilik = model.predict_proba(X.tail(1))[0][1] * 100

            # 3. GÖRSELLEŞTİRME
            st.subheader(f"📊 {secilen_hisse} Teknik Görünüm")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_200']].tail(200))

            # 4. AKILLI ANALİZ RAPORU
            st.markdown("---")
            st.markdown("### 🔍 AI Analiz Raporu")
            
            nedenler = []
            son_rsi = data['RSI'].iloc[-1]
            fiyat = data['Close'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]
            sma200 = data['SMA_200'].iloc[-1]

            if son_rsi < 35:
                nedenler.append(f"⚠️ **RSI ({son_rsi:.2f}):** Hisse aşırı satım bölgesine yakın. Teknik bir tepki gelebilir.")
            elif son_rsi > 65:
                nedenler.append(f"🔥 **RSI ({son_rsi:.2f}):** Hisse aşırı alım bölgesinde, kâr realizasyonu görülebilir.")
            
            if fiyat < sma20:
                nedenler.append(f"📉 **Trend:** Fiyat 20 günlük ortalamanın altında. Satıcılar baskın.")
            else:
                nedenler.append(f"🚀 **Trend:** Fiyat 20 günlük ortalamanın üzerinde. Momentum pozitif.")

            if vade == "Uzun Vade (6 Ay - 1 Yıl)":
                if fiyat < sma200:
                    nedenler.append(f"🧱 **Direnç:** Fiyat 200 günlük ana ortalamanın altında. Sabır gerekebilir.")
                else:
                    nedenler.append(f"🛡️ **Güven:** Fiyat 200 günlük ortalamanın üzerinde. Uzun vade trendi sağlıklı.")

            # Karar ve Nedenler
            col1, col2 = st.columns(2)
            with col1:
                if olasilik > 60:
                    st.success(f"📈 TAVSİYE: GÜÇLÜ AL (%{olasilik:.1f})")
                elif olasilik < 40:
                    st.error(f"📉 TAVSİYE: SAT / BEKLE (%{100-olasilik:.1f})")
                else:
                    st.warning(f"⚖️ TAVSİYE: NÖTR (%{olasilik:.1f})")
                
                for n in nedenler:
                    st.write(n)
            
            with col2:
                st.info(f"**Strateji:** {vade}\n\nAnaliz periyodu ve AI hedeflemesi seçtiğiniz vadeye göre optimize edildi.")

        else:
            st.error("Veri çekilemedi.")
