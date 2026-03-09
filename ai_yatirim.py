import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="AI Borsa Analisti", layout="wide")
st.title("📈 Metin Yüksel - Yapay Zeka Yatırım Paneli")

# KULLANICI GİRİŞİ
hisse = st.text_input("Hisse Kodu Girin (Örn: THYAO.IS, AAPL, ASUZU.IS):", "THYAO.IS").upper()

if st.button("Analiz Et"):
    with st.spinner('Veriler işleniyor...'):
        # 1. VERİ ÇEKME
        raw_data = yf.download(hisse, period="2y", interval="1d")
        
        if not raw_data.empty:
            # SÜTUN TEMİZLİĞİ (Hatanın Çözümü Burada!)
            data = raw_data.copy()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 2. İNDİKATÖRLER
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data.dropna(inplace=True)

            # 3. AI MODELİ
            X = data[['Close', 'Volume', 'SMA_20', 'RSI']]
            y = data['Target']
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            olasilik = model.predict_proba(X.tail(1))[0][1] * 100

            # 4. GÖRSELLEŞTİRME
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"{hisse} Fiyat ve Trend")
                # Grafiği açıkça tanımlıyoruz
                chart_data = data[['Close', 'SMA_20']].tail(100)
                st.line_chart(chart_data)
            
            with col2:
                st.subheader("RSI (Ateş Ölçer)")
                st.line_chart(data['RSI'].tail(100))

            # TAVSİYE KUTUSU
            if olasilik > 60:
                st.success(f"🚀 GÜÇLÜ AL SİNYALİ! (Artış İhtimali: %{olasilik:.1f})")
            elif olasilik < 40:
                st.error(f"⚠️ SATIŞ / BEKLE SİNYALİ (Düşüş İhtimali: %{100-olasilik:.1f})")
            else:
                st.warning(f"⚖️ BELİRSİZ / YATAY (İhtimal: %{olasilik:.1f})")
        else:
            st.error("Veri çekilemedi. Okul interneti engelliyor olabilir!")