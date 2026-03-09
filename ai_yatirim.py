import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SAYFA AYARLARI
st.set_page_config(page_title="Metin Yuksel - Yatirim Terminali", layout="wide")
st.title("📈 Metin Yüksel - Finansal Analiz ve Yatırım Terminali")

# 1. AYARLAR VE GİRİŞ PANELİ
st.sidebar.header("Analiz Parametreleri")
populer_hisseler = ["Kendim Yazacağım", "THYAO", "ASELS", "FROTO", "PGSUS", "EREGL", "SASA", "KCHOL", "TUPRS", "BIMAS"]
secim = st.sidebar.selectbox("Hisse Seçin veya Yazın:", populer_hisseler)

if secim == "Kendim Yazacağım":
    hisse_adi = st.sidebar.text_input("Hisse Kodu (Örn: GENKM, BTC-USD):", "THYAO").upper()
else:
    hisse_adi = secim

vade = st.sidebar.radio("Yatırım Vadesi:", ("Kısa Vade (1-15 Gün)", "Orta Vade (1-6 Ay)", "Uzun Vade (6 Ay+)"))

if st.sidebar.button("Analizi Gerçekleştir"):
    with st.spinner('Piyasa verileri analiz ediliyor...'):
        # BIST Kontrolü
        sembol = f"{hisse_adi}.IS" if len(hisse_adi) <= 5 and "-" not in hisse_adi else hisse_adi
        data = yf.download(sembol, period="5y", interval="1d")
        
        if not data.empty and len(data) > 50:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            # TEKNİK HESAPLAMALAR
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # AI MODELLEME (Vadeye Göre Dinamik)
            shift_map = {"Kısa Vade (1-15 Gün)": 5, "Orta Vade (1-6 Ay)": 22, "Uzun Vade (6 Ay+)": 60}
            target_shift = shift_map[vade]
            
            data['Target'] = (data['Close'].shift(-target_shift) > data['Close']).astype(int)
            features = ['Close', 'SMA_20', 'SMA_50', 'RSI']
            clean_df = data[features + ['Target']].dropna()
            
            if not clean_df.empty:
                X = clean_df[features]
                y = clean_df['Target']
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                olasilik = model.predict_proba(data[features].tail(1))[0][1] * 100

                # GÖRSELLEŞTİRME
                st.subheader(f"📊 {hisse_adi} Teknik Veriler")
                st.line_chart(data[['Close', 'SMA_20', 'SMA_50']].tail(180))

                # KARAR MEKANİZMASI (Dengeli)
                son_rsi = data['RSI'].iloc[-1]
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Mevcut Pozisyon Analizi")
                    if olasilik > 60:
                        st.success(f"✅ **KORU:** Yükseliş ihtimali güçlü (%{olasilik:.1f}).")
                    elif olasilik < 40:
                        st.error(f"⚠️ **AZALT:** Düşüş riski hakim (%{100-olasilik:.1f}).")
                    else:
                        st.warning("⚖️ **İZLE:** Net bir trend oluşumu bekleniyor.")

                with col2:
                    st.markdown("### Yeni Alım Stratejisi")
                    # AI ve RSI dengesi
                    if olasilik > 55 and son_rsi < 60:
                        st.success("🎯 **UYGUN:** AI desteği ve teknik alan müsait.")
                    elif son_rsi < 30:
                        st.info("🔎 **TEPKİ BEKLENİYOR:** Aşırı satım bölgesinde, kademeli takip edilebilir.")
                    else:
                        st.write("⌛ **BEKLE:** Alım için daha net bir teknik onay gerekiyor.")

                st.caption(f"Veriler Yahoo Finance üzerinden {vade} stratejisine göre işlenmiştir.")
            else:
                st.error("Yeterli veri seti oluşmadı.")
        else:
            st.error("Hisse verisi bulunamadı veya çok yeni.")
