import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Metin Yüksel AI v6.0", layout="wide")
st.title("🏹 Metin Yüksel - Korkusuz Strateji Terminali v6.0")

# 1. HİSSE VE VADE SEÇİMİ
st.sidebar.header("⚙️ Ayarlar")
hisse_kod = st.sidebar.text_input("Hisse Kodu (Örn: THYAO, BTC-USD, ASUZU):", "THYAO").upper()
vade = st.sidebar.selectbox("Tahmin Vadesi:", ["5 Günlük (Hızlı)", "22 Günlük (Orta)"])

if st.button("Teknik Analizi Patlat"):
    with st.spinner('Piyasa verileri AI ile çarpıştırılıyor...'):
        sembol = f"{hisse_kod}.IS" if len(hisse_kod) <= 5 and "-" not in hisse_kod else hisse_kod
        data = yf.download(sembol, period="2y", interval="1d")
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            # --- TEKNİK HESAPLAMALAR ---
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # --- AI MODELLEME (CESUR MOD) ---
            shift_days = 5 if "5" in vade else 22
            data['Target'] = (data['Close'].shift(-shift_days) > data['Close']).astype(int)
            features = ['Close', 'SMA_20', 'SMA_50', 'RSI']
            clean_df = data[features + ['Target']].dropna()
            
            X = clean_df[features]
            y = clean_df['Target']
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X, y)
            
            olasilik = model.predict_proba(data[features].tail(1))[0][1] * 100

            # --- GÖRSEL VE KARAR ---
            son_fiyat = data['Close'].iloc[-1]
            son_rsi = data['RSI'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]
            
            st.subheader(f"📊 {hisse_kod} Analiz Tablosu")
            st.line_chart(data[['Close', 'SMA_20', 'SMA_50']].tail(120))

            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 💼 Mevcut Pozisyon")
                # KARAR MEKANİZMASI (HİBRİT)
                if olasilik > 50 or (son_rsi < 30):
                    st.success(f"🚀 **TUT / EKLE:** AI Güveni: %{olasilik:.1f} | RSI: {son_rsi:.1f}")
                else:
                    st.error(f"📉 **BEKLE / AZALT:** AI Güveni Düşük: %{olasilik:.1f}")

            with c2:
                st.markdown("### 💰 Yeni Alım Kararı")
                # Teknik Filtreler AI'yı Dövebilir
                if son_rsi < 28:
                    st.success("💎 **KESİN ALIM BÖLGESİ:** AI ne derse desin, hisse teknik olarak aşırı ucuz (DİP).")
                elif olasilik > 55 and son_fiyat > sma20:
                    st.success("✅ **ALIM UYGUN:** Hem AI hem trend yükselişi onaylıyor.")
                elif olasilik > 50:
                    st.warning("🔎 **KADEMELİ AL:** AI olumlu sinyal veriyor, parça parça girilebilir.")
                else:
                    st.info("⌛ **SABRET:** Henüz net bir dönüş sinyali yok.")

            st.markdown("---")
            st.caption(f"Not: Bu sistem AI tahmini (%{olasilik:.1f}) ile teknik indikatörleri harmanlayarak karar verir.")
        else:
            st.error("Hisse kodu hatalı!")
