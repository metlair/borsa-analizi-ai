import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Metin Yüksel AI Terminal v5.0", layout="wide")
st.title("🛡️ Metin Yüksel - AI Strateji Terminali v5.0")

# 1. GELİŞMİŞ HİSSE ARAMA
bist_listesi = ["THYAO", "ASELS", "FROTO", "PGSUS", "EREGL", "SASA", "KCHOL", "SISE", "AKBNK", "TUPRS", "BIMAS", "HEKTS", "DOAS", "ASTOR", "KONTR", "ALARK", "ODAS", "ARCLK", "PETKM"]
secilen_hisse = st.sidebar.selectbox("Hisse Seçin veya Yazın:", ["MANUEL GİRİŞ"] + bist_listesi)

if secilen_hisse == "MANUEL GİRİŞ":
    hisse_kod = st.sidebar.text_input("Hisse Kodu (Örn: BTC-USD, ASUZU):", "THYAO").upper()
else:
    hisse_kod = secilen_hisse

vade = st.sidebar.radio("Tahmin Vadesi:", ("5 Günlük", "22 Günlük (1 Ay)", "Uzun Vade"))

if st.button("Derin Teknik Analizi Başlat"):
    with st.spinner('Piyasa psikolojisi ve indikatörler taranıyor...'):
        sembol = f"{hisse_kod}.IS" if len(hisse_kod) <= 5 and "-" not in hisse_kod else hisse_kod
        data = yf.download(sembol, period="5y", interval="1d")
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            # --- MÜHENDİSLİK ÖZELLİKLERİ (FEATURES) ---
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            # Momentum (Fiyatın 10 gün önceki fiyata oranı)
            data['Momentum'] = data['Close'] / data['Close'].shift(10)
            # Volatilite
            data['Volatilite'] = data['Close'].rolling(window=10).std()

            # AI EĞİTİMİ
            target_shift = 5 if "5" in vade else 22
            data['Target'] = (data['Close'].shift(-target_shift) > data['Close']).astype(int)
            
            # Veri Hazırlama (Sadece gerekli olanları al)
            features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'Momentum', 'Volatilite']
            clean_df = data[features + ['Target']].dropna()
            
            X = clean_df[features]
            y = clean_df['Target']
            
            model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
            model.fit(X, y)
            
            # Tahmin
            current_state = data[features].tail(1)
            olasilik = model.predict_proba(current_state)[0][1] * 100
            
            # GÖRSEL RAPOR
            st.subheader(f"📊 {hisse_kod} Teknik Yol Haritası")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_50']].tail(150))
            
            c1, c2 = st.columns(2)
            son_rsi = data['RSI'].iloc[-1]
            son_fiyat = data['Close'].iloc[-1]

            with c1:
                st.info("### 📋 Portföy Yönetimi")
                if olasilik > 52:
                    st.success(f"⏫ **POZİSYON KORU:** AI %{olasilik:.1f} ihtimalle yukarı yönlü potansiyel görüyor.")
                elif son_rsi < 35:
                    st.warning(f"⚖️ **TEPKİ BEKLE:** Düşüş ihtimali var ama RSI ({son_rsi:.1f}) dipte. Satmak için geç olabilir.")
                else:
                    st.error(f"⏬ **AZALT:** %{100-olasilik:.1f} ihtimalle baskı sürebilir.")

            with c2:
                st.info("### 🏹 Alım Stratejisi")
                if olasilik > 55 and son_rsi < 65:
                    st.success("🎯 **ALIM FIRSATI:** AI ve teknik veriler alımı destekliyor.")
                elif son_rsi < 25:
                    st.success("💎 **KELEPİR:** AI karamsar olsa da aşırı satım var. Kademeli toplanabilir.")
                elif son_rsi > 75:
                    st.error("🚫 **ALMA:** Fiyat çok şişti, düzeltme beklemek daha güvenli.")
                else:
                    st.write("🔎 **GÖZLEM:** Net bir alım bölgesi oluşmadı.")

            st.markdown("---")
            st.write(f"**Mühendislik Notu:** Bu analiz {vade} için optimize edilmiş **RandomForest** modeli tarafından üretilmiştir.")
        else:
            st.error("Veri çekilemedi!")
