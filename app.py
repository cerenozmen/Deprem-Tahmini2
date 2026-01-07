import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(
    page_title="İstanbul Deprem Tahmin Modeli",
    page_icon="mag",
    layout="wide"
)

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
    "Sa
