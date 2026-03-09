import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="AI Yatırım Terminali", layout="wide")
st.title("🚀 Metin Yüksel - Profesyonel Yatırım Terminali")

# 1. BIST 100 LİSTESİ (Kullanıcı Dostu Seçim)
bist_hisseleri = ["THYAO", "ASELS", "FROTO", "PGSUS", "EREGL", "SASA", "KCHOL", "SISE", "AKBNK", "GARAN", "TUPRS", "ISCTR"]
secilen_hisse = st.selectbox("Analiz Edilecek Hisseyi Seçin:", bist_hisseleri)
vade = st.radio("Yatırım Stratejiniz Nedir?", ("Kısa Vade (1-15 Gün)", "Uzun Vade (6 Ay - 1 Yıl)"))

if st.button("Teknik Analizi Başlat"):
    with st.spinner('Derin analiz yapılıyor...'):
        hisse_kodu = f"{secilen_hisse}.IS"
        # Vadeye göre veri periyodu seçimi
        periyot = "1y" if vade == "Kısa Vade (1-15 Gün)" else "5y"
        data = yf.download(hisse_kodu, period=periyot, interval="1d")
        
        if not data.empty:
            # SÜTUN TEMİZLİĞİ
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # 2. TEKNİK HESAPLAMALAR
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI MODELİ (Hedef vadeye göre eğitim)
            shift_days = 5 if vade == "Kısa Vade (1-15 Gün)" else 60
            data['Target'] = (data['Close'].shift(-shift_days) > data['Close']).astype(int)
            data.dropna(inplace=True)
            
            X = data[['Close', 'SMA_20', 'RSI']]
            y = data['Target']
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            
            olasilik = model.predict_proba(X.tail(1))[0][1] * 100

            # 3. GÖRSELLEŞTİRME VE RAPOR
            st.subheader(f"📊 {secilen_hisse} Teknik Görünüm")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_200']].tail(200))

            # 4. TEKNİK GEREKÇELER (Neden Analizi)
            st.markdown("### 🔍 Neden Bu Karar Verildi?")
            son_rsi = data['RSI'].iloc[-1]
            fiyat = data['Close'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]

            if olasilik > 60:
                st.success(f"📈 AI TAVSİYESİ: AL SİNYALİ (%{olasilik:.1f})")
            elif olasilik < 40:
                st.error(f"📉 AI TAVSİYESİ: SAT/BEKLE SİNYALİ (%{100-olasilik:.1f})")
            else:
                st.warning(f"⚖️ AI TAVSİYESİ: NÖTR / İZLE SİNYALİ (%{olasilik:.1f})")

            # Teknik Detaylar
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**RSI Değeri:** {son_rsi:.2f}")
                st.write("**SMA 20 Durumu:** " + ("Fiyat üzerinde (Olumlu)" if fiyat > sma20 else "Fiyat altında (Olumsuz)"))
            with col2:
                st.info(f"Bu analiz, seçtiğiniz **{vade}** stratejisine göre optimize edilmiştir.")

        else:
            st.error("Veri çekilemedi.")
