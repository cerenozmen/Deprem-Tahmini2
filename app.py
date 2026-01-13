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

# --- İLÇE -> (lat, lon) HARİTASI (yaklaşık merkez koordinatlar) ---
ISTANBUL_DISTRICTS = {
    "Adalar": (40.874, 29.128),
    "Arnavutköy": (41.185, 28.740),
    "Ataşehir": (40.992, 29.124),
    "Avcılar": (40.979, 28.721),
    "Bağcılar": (41.034, 28.856),
    "Bahçelievler": (41.002, 28.861),
    "Bakırköy": (40.976, 28.872),
    "Başakşehir": (41.094, 28.802),
    "Bayrampaşa": (41.045, 28.900),
    "Beşiktaş": (41.043, 29.005),
    "Beykoz": (41.135, 29.090),
    "Beylikdüzü": (40.990, 28.641),
    "Beyoğlu": (41.038, 28.977),
    "Büyükçekmece": (41.020, 28.580),
    "Çatalca": (41.143, 28.463),
    "Çekmeköy": (41.035, 29.175),
    "Esenler": (41.044, 28.873),
    "Esenyurt": (41.035, 28.676),
    "Eyüpsultan": (41.066, 28.933),
    "Fatih": (41.018, 28.949),
    "Gaziosmanpaşa": (41.060, 28.916),
    "Güngören": (41.022, 28.871),
    "Kadıköy": (40.990, 29.029),
    "Kağıthane": (41.079, 28.969),
    "Kartal": (40.890, 29.190),
    "Küçükçekmece": (40.996, 28.775),
    "Maltepe": (40.936, 29.154),
    "Pendik": (40.875, 29.236),
    "Sancaktepe": (41.003, 29.231),
    "Sarıyer": (41.166, 29.050),
    "Silivri": (41.073, 28.246),
    "Sultanbeyli": (40.968, 29.270),
    "Sultangazi": (41.107, 28.868),
    "Şile": (41.175, 29.612),
    "Şişli": (41.060, 28.987),
    "Tuzla": (40.817, 29.303),
    "Ümraniye": (41.024, 29.124),
    "Üsküdar": (41.023, 29.016),
    "Zeytinburnu": (40.994, 28.902),
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
            f"Model dosyaları bulunamadı! Lütfen .joblib dosyalarının 'app.py' ile aynı klasörde olduğundan emin olun. Hata: {e}"
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

def district_to_latlon(district_name: str):
    return ISTANBUL_DISTRICTS.get(district_name, (41.000, 29.000))

def clamp(x, lo=2.0, hi=8.0):
    """Demo güvenliği: Mw değerlerini makul aralıkta tutar."""
    return np.clip(x, lo, hi)

def apply_calibration(preds, log_energy, b_value):
    """
    Demo amaçlı kalibrasyon:
    - Enerji ↑ => Mw ↑
    - b ↑      => Mw ↓
    Katsayıları özellikle 'ciddi senaryoda ~6' görünsün diye biraz güçlü tuttum.
    """
    preds = np.array(preds, dtype=float)
    preds_adj = preds + 0.70 * (log_energy - 8.0) - 0.35 * (b_value - 1.0)
    return preds_adj

# --- ARAYÜZ ---
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri kullanarak deprem büyüklüğü tahmini ve risk analizi yapar.")

tab1, tab2 = st.tabs(["📉 Deprem Büyüklüğü Tahmini (Regresyon)", "⚠️ Bölgesel Risk Analizi (Sınıflandırma)"])

# ---------------------------------------------------------
# TAB 1: REGRESYON
# ---------------------------------------------------------
with tab1:
    st.header("Senaryo Bazlı Büyüklük Tahmini")
    st.info("Aşağıdaki parametreleri girerek olası bir depremin tahmini büyüklüğünü (Magnitude) hesaplayın.")

    if rf_reg is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📍 Konum ve Zaman")
            district = st.selectbox("İlçe Seçin", options=sorted(ISTANBUL_DISTRICTS.keys()))
            input_lat, input_lon = district_to_latlon(district)
            st.caption(f"İlçe merkez koordinatı (otomatik): {input_lat:.3f}, {input_lon:.3f}")

            input_depth = st.number_input("Derinlik (km)", value=10.0, min_value=0.0)
            input_date = st.date_input("Başlangıç Tarihi (7 günlük tahmin için)", datetime.date.today())

        with col2:
            st.subheader("🌋 Sismik Parametreler")
            input_fault_dist = st.number_input("Fay Hattına Uzaklık (km)", value=5.0, min_value=0.0)
            input_b_value = st.number_input("b-değeri (Sismik aktivite eğimi)", value=1.0, min_value=0.2, max_value=2.5, step=0.05)
            input_log_energy = st.number_input("Log Enerji", value=9.0, min_value=0.0, step=0.1)

        with col3:
            st.subheader("⚡ Enerji İstatistikleri (Detay)")
            input_e30 = st.number_input("30 Günlük Enerji", value=10000.0, min_value=0.0)
            input_e90 = st.number_input("90 Günlük Enerji", value=50000.0, min_value=0.0)

            with st.expander("Gelişmiş Enerji Parametreleri"):
                input_er30 = st.number_input("Enerji Hızı (30 Gün)", value=100.0, min_value=0.0)
                input_er90 = st.number_input("Enerji Hızı (90 Gün)", value=100.0, min_value=0.0)

                input_log_e30 = np.log1p(input_e30)
                input_log_e90 = np.log1p(input_e90)
                input_log_er30 = np.log1p(input_er30)
                input_log_er90 = np.log1p(input_er90)

        st.divider()
        use_calibration = st.checkbox("Kalibrasyon (demo) uygula: ciddi senaryolar daha yüksek görünsün", value=True)
        st.caption("Kalibrasyon açıkken: Ham tahmin + enerji/b düzeltmesi uygulanır. Kapalıyken: ham model çıktısı gösterilir.")

        if st.button("7 Günlük Büyüklük Tahmini", type="primary"):
            try:
                dates = [input_date + datetime.timedelta(days=i) for i in range(7)]
                rows = []
                for d in dates:
                    date_feats = derive_date_features(d)
                    rows.append({
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
                        "month": date_feats["month"],
                        "dow": date_feats["dow"],
                        "dayofyear": date_feats["dayofyear"]
                    })

                input_df = pd.DataFrame(rows)

                # Ham tahmin
                preds = rf_reg.predict(input_df)
                preds = np.array(preds, dtype=float)

                # Kalibre tahmin
                if use_calibration:
                    preds_adj = apply_calibration(preds, input_log_energy, input_b_value)
                    preds_final = clamp(preds_adj, 2.0, 8.0)
                else:
                    preds_final = clamp(preds, 2.0, 8.0)

                out = pd.DataFrame({
                    "Tarih": dates,
                    "Ham Mw": np.round(preds, 2),
                    "Kalibre Mw": np.round(preds_final, 2),
                })

                st.success("7 günlük tahmin başarıyla tamamlandı!")
                st.dataframe(out, use_container_width=True)

                max_pred = float(np.max(preds_final))
                if max_pred >= 7.0:
                    st.error(f"Hafta içi en yüksek tahmin: {max_pred:.2f} → KRİTİK / YIKICI")
                elif max_pred >= 5.0:
                    st.warning(f"Hafta içi en yüksek tahmin: {max_pred:.2f} → CİDDİ / ORTA")
                else:
                    st.success(f"Hafta içi en yüksek tahmin: {max_pred:.2f} → HAFİF / DÜŞÜK")

                chart_df = out.set_index("Tarih")[["Ham Mw", "Kalibre Mw"]]
                st.line_chart(chart_df)

                st.caption(
                    f"Ham min/max: {float(np.min(preds)):.2f} / {float(np.max(preds)):.2f} | "
                    f"Kalibre min/max: {float(np.min(preds_final)):.2f} / {float(np.max(preds_final)):.2f}"
                )

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.write("Lütfen modelin feature isim/sıralamasının kod ile eşleştiğinden emin olun.")

# ---------------------------------------------------------
# TAB 2: SINIFLANDIRMA
# ---------------------------------------------------------
with tab2:
    st.header("Bölgesel Deprem Olasılığı (M ≥ 3.0)")
    st.write("Seçilen bölge ve geçmiş aktivite verilerine göre deprem olma olasılığını hesaplar.")

    if rf_clf is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Konum Bilgileri")
            c_district = st.selectbox("İlçe Seçin", options=sorted(ISTANBUL_DISTRICTS.keys()), key="c_district")
            c_lat, c_lon = district_to_latlon(c_district)
            st.caption(f"İlçe merkez koordinatı (otomatik): {c_lat:.3f}, {c_lon:.3f}")

            lat_bin = np.floor(c_lat / 0.1) * 0.1
            lon_bin = np.floor(c_lon / 0.1) * 0.1
            st.write(f"Hesaplanan Hücre: {lat_bin:.1f}, {lon_bin:.1f}")

        with c2:
            st.subheader("Geçmiş 30 Günlük Aktivite")
            st.caption("Bu değerler normalde veri tabanından otomatik çekilir. Senaryo için manuel giriniz.")

            roll30_count = st.number_input("Son 30 gündeki deprem sayısı", value=5.0, min_value=0.0, step=1.0)
            roll30_maxmag = st.number_input("Son 30 gündeki maks. büyüklük", value=3.5, min_value=0.0, step=0.1)
            roll30_meanmag = st.number_input("Son 30 gündeki ort. büyüklük", value=2.5, min_value=0.0, step=0.1)
            roll30_depth = st.number_input("Son 30 gündeki ort. derinlik", value=10.0, min_value=0.0, step=0.5)

            roll30_energy = 1000.0
            roll30_energy_rate = 10.0

        c_date_input = st.date_input("Başlangıç Tarihi (7 günlük risk için)", datetime.date.today(), key="c_date")

        if st.button("7 Günlük Risk Hesapla", type="primary"):
            try:
                dates = [c_date_input + datetime.timedelta(days=i) for i in range(7)]
                rows = []
                for d in dates:
                    feats = derive_date_features(d)
                    rows.append({
                        "lat_bin": lat_bin,
                        "lon_bin": lon_bin,
                        "roll30_count": roll30_count,
                        "roll30_maxmag": roll30_maxmag,
                        "roll30_meanmag": roll30_meanmag,
                        "roll30_depth": roll30_depth,
                        "roll30_energy_30d": roll30_energy,
                        "roll30_energy_rate_30d": roll30_energy_rate,
                        "month": feats["month"],
                        "dow": feats["dow"],
                        "dayofyear": feats["dayofyear"]
                    })

                clf_input = pd.DataFrame(rows)
                probs = rf_clf.predict_proba(clf_input)[:, 1]

                out = pd.DataFrame({
                    "Tarih": dates,
                    "Olasılık (M≥3.0)": np.round(probs, 4),
                    "Yüzde": np.round(probs * 100, 2)
                })

                st.success("7 günlük risk analizi tamamlandı!")
                st.dataframe(out, use_container_width=True)

                max_prob = float(np.max(probs))
                st.divider()
                st.subheader(f"Hafta içi en yüksek olasılık: %{max_prob*100:.2f}")
                st.progress(max_prob)

                if max_prob > 0.7:
                    st.error("Yüksek Risk!")
                elif max_prob > 0.4:
                    st.warning("Orta Risk")
                else:
                    st.success("Düşük Risk")

                chart_df = out.set_index("Tarih")[["Yüzde"]]
                st.line_chart(chart_df)

            except Exception as e:
                st.error(f"Sınıflandırma hatası: {e}")

st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
