import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="Metin Yüksel AI Terminal", layout="wide")
st.title("🚀 Metin Yüksel - Profesyonel Yatırım Terminali v4.0")

# 1. HİSSE GİRİŞİ (Serbest Yazım + Öneri)
populer_hisseler = ["THYAO", "ASELS", "FROTO", "PGSUS", "EREGL", "SASA", "KCHOL", "SISE", "AKBNK", "TUPRS", "BIMAS", "HEKTS"]
st.sidebar.header("⚙️ Analiz Ayarları")
hisse_input = st.sidebar.selectbox("Popüler Hisseler:", ["Kendim Yazacağım"] + populer_hisseler)

if hisse_input == "Kendim Yazacağım":
    secilen_hisse = st.sidebar.text_input("Hisse Kodu Yazın (Örn: BTC-USD, AAPL, ASUZU):", "THYAO").upper()
else:
    secilen_hisse = hisse_input

vade = st.sidebar.radio("Strateji:", ("Kısa Vade (1-15 Gün)", "Orta Vade (1-6 Ay)", "Uzun Vade (1 Yıl+)"))

if st.button("Kapsamlı Analizi Başlat"):
    with st.spinner('Analiz ediliyor...'):
        # BIST hissesi mi kontrolü
        hisse_kodu = f"{secilen_hisse}.IS" if len(secilen_hisse) <= 5 and "-" not in secilen_hisse else secilen_hisse
        
        periyot = "2y" if "Kısa" in vade else "5y"
        data = yf.download(hisse_kodu, period=periyot, interval="1d")
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # TEKNİK VERİLER
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI EĞİTİMİ (Daha hassas hale getirildi)
            shift_days = 5 if "Kısa" in vade else 22
            data['Target'] = (data['Close'].shift(-shift_days) > data['Close']).astype(int)
            data.dropna(inplace=True)
            
            # Daha fazla özellik ekleyerek AI'nın sadece fiyata bakmamasını sağladık
            X = data[['Close', 'SMA_20', 'SMA_50', 'RSI']]
            y = data['Target']
            model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5) # Aşırı öğrenmeyi azalttık
            model.fit(X, y)
            olasilik = model.predict_proba(X.tail(1))[0][1] * 100

            # GÖRSELLEŞTİRME
            st.subheader(f"📊 {secilen_hisse} Analiz Paneli")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_50']].tail(200))

            # STRATEJİ RAPORU
            col1, col2 = st.columns(2)
            son_rsi = data['RSI'].iloc[-1]
            
            with col1:
                st.markdown("### 💼 Mevcut Pozisyon")
                if olasilik > 55:
                    st.success(f"📈 **TUT:** AI %{olasilik:.1f} ihtimalle yükseliş bekliyor.")
                elif olasilik < 45:
                    st.error(f"📉 **AZALT:** AI %{100-olasilik:.1f} ihtimalle düşüş bekliyor.")
                else:
                    st.warning("⚖️ **BEKLE:** Net bir yön tayin edilemiyor.")

            with col2:
                st.markdown("### 💰 Yeni Giriş")
                if olasilik > 55 and son_rsi < 65:
                    st.success(f"✅ **ALIM UYGUN:** Teknik göstergeler girişi destekliyor.")
                elif son_rsi > 75:
                    st.error("❌ **ALMA:** Hisse aşırı şişmiş (RSI > 75). Düzeltme gelebilir.")
                else:
                    st.info("🔎 **GÖZLEM:** Alım için daha net bir sinyal beklenmeli.")

            st.metric("AI Güven Endeksi", f"%{olasilik:.1f}", delta=f"{olasilik-50:.1f}")

        else:
            st.error("Hisse bulunamadı! Lütfen kodu doğru yazdığınızdan emin olun.")
