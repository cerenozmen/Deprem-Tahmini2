import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Sayfa Ayarları
st.set_page_config(
    page_title="İstanbul Deprem Tahmin Modeli",
    page_icon="mag",
    layout="wide"
)

# -----------------------------
# İLÇE -> KOORDİNAT HARİTASI
# -----------------------------
DISTRICT_COORDS = {
    "Adalar": (40.8680, 29.1290),
    "Arnavutköy": (41.1846, 28.7403),
    "Ataşehir": (40.9929, 29.1247),
    "Avcılar": (40.9792, 28.7214),
    "Bağcılar": (41.0390, 28.8567),
    "Bahçelievler": (40.9977, 28.8506),
    "Bakırköy": (40.9819, 28.8728),
    "Başakşehir": (41.0930, 28.8020),
    "Bayrampaşa": (41.0404, 28.9025),
    "Beşiktaş": (41.0430, 29.0094),
    "Beykoz": (41.1340, 29.0947),
    "Beylikdüzü": (40.9760, 28.6370),
    "Beyoğlu": (41.0369, 28.9847),
    "Büyükçekmece": (41.0207, 28.5850),
    "Çatalca": (41.1426, 28.4620),
    "Çekmeköy": (41.0404, 29.1736),
    "Esenler": (41.0465, 28.8764),
    "Esenyurt": (41.0343, 28.6801),
    "Eyüpsultan": (41.0481, 28.9334),
    "Fatih": (41.0186, 28.9396),
    "Gaziosmanpaşa": (41.0585, 28.9120),
    "Güngören": (41.0171, 28.8803),
    "Kadıköy": (40.9917, 29.0275),
    "Kağıthane": (41.0850, 28.9667),
    "Kartal": (40.8905, 29.1857),
    "Küçükçekmece": (40.9978, 28.7896),
    "Maltepe": (40.9350, 29.1550),
    "Pendik": (40.8775, 29.2526),
    "Sancaktepe": (41.0024, 29.2313),
    "Sarıyer": (41.1680, 29.0576),
    "Silivri": (41.0731, 28.2460),
    "Sultanbeyli": (40.9689, 29.2629),
    "Sultangazi": (41.1037, 28.8661),
    "Şile": (41.1746, 29.6111),
    "Şişli": (41.0602, 28.9877),
    "Tuzla": (40.8161, 29.3006),
    "Ümraniye": (41.0247, 29.1245),
    "Üsküdar": (41.0227, 29.0235),
    "Zeytinburnu": (40.9944, 28.9042)
}

# --- MODELLERİN YÜKLENMESİ ---
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load('rf_reg_deprem_buyukluk.joblib')
        clf_model = joblib.load('rf_clf_deprem_olasilik.joblib')
        return reg_model, clf_model
    except FileNotFoundError as e:
        st.error(
            "Model dosyaları bulunamadı! Lütfen .joblib dosyalarının 'app.py' ile aynı klasörde olduğundan emin olun.\n"
            f"Hata: {e}"
        )
        return None, None

rf_reg, rf_clf = load_models()

# --- YARDIMCI FONKSİYONLAR ---
def derive_date_features(selected_date):
    return {
        "month": selected_date.month,
        "dow": selected_date.weekday(),
        "dayofyear": selected_date.timetuple().tm_yday
    }

def build_reg_input_rows(
    base_params: dict,
    start_date: datetime.date,
    days: int = 7
) -> pd.DataFrame:
    """Başlangıç tarihinden itibaren N gün için model input satırlarını üretir."""
    rows = []
    for i in range(days):
        d = start_date + datetime.timedelta(days=i)
        feats = derive_date_features(d)
        row = dict(base_params)
        row.update({
            "month": feats["month"],
            "dow": feats["dow"],
            "dayofyear": feats["dayofyear"],
            "date": d  # sadece ekranda göstermek için
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Model inputunda "date" feature değil — predict öncesi çıkaracağız.
    return df

def severity_label(pred: float) -> str:
    if pred >= 7.0:
        return "KRİTİK / YIKICI"
    elif pred >= 5.0:
        return "CİDDİ / ORTA"
    return "HAFİF / DÜŞÜK"

# --- ARAYÜZ ---
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri kullanarak deprem büyüklüğü tahmini ve risk analizi yapar.")

tab1, tab2 = st.tabs(["📉 Deprem Büyüklüğü Tahmini (Regresyon)", "⚠️ Bölgesel Risk Analizi (Sınıflandırma)"])

# ---------------------------------------------------------
# TAB 1: REGRESYON (BÜYÜKLÜK TAHMİNİ) — 1 HAFTALIK
# ---------------------------------------------------------
with tab1:
    st.header("Senaryo Bazlı Büyüklük Tahmini")
    st.info("Parametreleri girin; seçtiğiniz tarihten itibaren 7 günlük tahmini büyüklük (Mw) üretelim.")

    if rf_reg is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📍 Konum ve Zaman")
            district = st.selectbox(
                "İlçe Seçiniz",
                sorted(list(DISTRICT_COORDS.keys()))
            )
            input_lat, input_lon = DISTRICT_COORDS[district]
            st.caption(f"Seçilen ilçe: **{district}** | Koordinat: **{input_lat:.4f}, {input_lon:.4f}**")

            input_depth = st.number_input("Derinlik (km)", value=10.0, min_value=0.0)
            start_date = st.date_input("Başlangıç Tarihi", datetime.date.today())
            days_to_forecast = st.slider("Kaç gün tahmin?", min_value=2, max_value=14, value=7)

        with col2:
            st.subheader("🌋 Sismik Parametreler")
            input_fault_dist = st.number_input("Fay Hattına Uzaklık (km)", value=5.0)
            input_b_value = st.number_input("b-değeri (Sismik aktivite eğimi)", value=1.0)
            input_log_energy = st.number_input("Log Enerji", value=9.0)

        with col3:
            st.subheader("⚡ Enerji İstatistikleri (Detay)")
            input_e30 = st.number_input("30 Günlük Enerji", value=10000.0)
            input_e90 = st.number_input("90 Günlük Enerji", value=50000.0)

            with st.expander("Gelişmiş Enerji Parametreleri"):
                input_er30 = st.number_input("Enerji Hızı (30 Gün)", value=100.0)
                input_er90 = st.number_input("Enerji Hızı (90 Gün)", value=100.0)

                input_log_e30 = np.log1p(input_e30)
                input_log_e90 = np.log1p(input_e90)
                input_log_er30 = np.log1p(input_er30)
                input_log_er90 = np.log1p(input_er90)

        if st.button("1 Haftalık Tahmin Üret", type="primary"):
            try:
                # Tarih hariç tüm sabit parametreler
                base_params = {
                    "lat": input_lat,
                    "lon": input_lon,
                    "depth_km": input_depth,
                    "fault_distance": input_fault_dist,
                    "b_value": input_b_value,
                    "log_energy": input_log_energy,
                    "energy_30d": input_e30,
                    "energy_rate_30d": input_er30,
                    "energy_90d": input_e90,
                    "energy_rate_90d": input_er90,
                    "log_energy_30d": input_log_e30,
                    "log_energy_90d": input_log_e90,
                    "log_energy_rate_30d": input_log_er30,
                    "log_energy_rate_90d": input_log_er90,
                }

                df_inputs = build_reg_input_rows(base_params, start_date, days=days_to_forecast)

                # Modelin beklediği sütunlar: "date" hariç
                X = df_inputs.drop(columns=["date"])

                preds = rf_reg.predict(X)

                results = pd.DataFrame({
                    "Tarih": df_inputs["date"],
                    "Tahmini Mw": np.round(preds, 2),
                    "Durum": [severity_label(p) for p in preds]
                })

                st.success("Haftalık tahmin üretildi!")
                st.dataframe(results, use_container_width=True)

                st.subheader("📈 Günlük Tahmin Grafiği")
                chart_df = results.set_index("Tarih")[["Tahmini Mw"]]
                st.line_chart(chart_df)

                st.subheader("📌 Özet")
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Maks Mw", f"{np.max(preds):.2f}")
                with colB:
                    st.metric("Ortalama Mw", f"{np.mean(preds):.2f}")
                with colC:
                    st.metric("Min Mw", f"{np.min(preds):.2f}")

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.write("Modelin feature isimleri/sırası ile kodun eşleştiğinden emin olun.")

# ---------------------------------------------------------
# TAB 2: SINIFLANDIRMA (RİSK ANALİZİ)
# ---------------------------------------------------------
with tab2:
    st.header("Bölgesel Deprem Olasılığı (M ≥ 3.0)")
    st.write("Seçilen bölge ve geçmiş aktivite verilerine göre deprem olma olasılığını hesaplar.")

    if rf_clf is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Konum Bilgileri")
            district_c = st.selectbox(
                "İlçe Seçiniz",
                sorted(list(DISTRICT_COORDS.keys())),
                key="district_c"
            )
            c_lat, c_lon = DISTRICT_COORDS[district_c]

            lat_bin = np.floor(c_lat / 0.1) * 0.1
            lon_bin = np.floor(c_lon / 0.1) * 0.1

            st.write(f"Seçilen ilçe: **{district_c}**")
            st.write(f"Hesaplanan Hücre: **{lat_bin:.1f}, {lon_bin:.1f}**")
            st.caption(f"Kullanılan koordinatlar: {c_lat:.4f}, {c_lon:.4f}")

        with c2:
            st.subheader("Geçmiş 30 Günlük Aktivite")
            st.caption("Bu değerler normalde veri tabanından otomatik çekilir. Senaryo için manuel giriniz.")

            roll30_count = st.number_input("Son 30 gündeki deprem sayısı", value=5.0)
            roll30_maxmag = st.number_input("Son 30 gündeki maks. büyüklük", value=3.5)
            roll30_meanmag = st.number_input("Son 30 gündeki ort. büyüklük", value=2.5)
            roll30_depth = st.number_input("Son 30 gündeki ort. derinlik", value=10.0)

            roll30_energy = 1000.0
            roll30_energy_rate = 10.0

        c_date_input = st.date_input("Analiz Tarihi", datetime.date.today(), key="c_date")

        if st.button("Risk Hesapla", type="primary"):
            try:
                c_date_feats = derive_date_features(c_date_input)

                clf_input = pd.DataFrame([{
                    "lat_bin": lat_bin,
                    "lon_bin": lon_bin,
                    "roll30_count": roll30_count,
                    "roll30_maxmag": roll30_maxmag,
                    "roll30_meanmag": roll30_meanmag,
                    "roll30_depth": roll30_depth,
                    "roll30_energy_30d": roll30_energy,
                    "roll30_energy_rate_30d": roll30_energy_rate,
                    "month": c_date_feats['month'],
                    "dow": c_date_feats['dow'],
                    "dayofyear": c_date_feats['dayofyear']
                }])

                prob = rf_clf.predict_proba(clf_input)[0][1]

                st.divider()
                st.subheader(f"M ≥ 3.0 Deprem Olasılığı: %{prob*100:.2f}")
                st.progress(prob)

                if prob > 0.7:
                    st.error("Yüksek Risk!")
                elif prob > 0.4:
                    st.warning("Orta Risk")
                else:
                    st.success("Düşük Risk")

            except Exception as e:
                st.error(f"Sınıflandırma hatası: {e}")

st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
