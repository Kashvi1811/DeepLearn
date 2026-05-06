from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
try:
    from PIL import Image, ImageFilter, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
from scipy import ndimage
from sklearn.datasets import make_circles, make_classification, make_moons, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    HAS_TF = True
except Exception:
    tf = None
    keras = None
    layers = None
    HAS_TF = False


try:
    from streamlit.runtime import exists as streamlit_runtime_exists

    IN_STREAMLIT_RUNTIME = streamlit_runtime_exists()
except Exception:
    IN_STREAMLIT_RUNTIME = False


if not IN_STREAMLIT_RUNTIME:
    print("Run this app with: streamlit run app.py")
    raise SystemExit(0)


st.set_page_config(page_title="DeepLearn", layout="wide", page_icon="🧠", initial_sidebar_state="expanded")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

        :root {
            --bg: #07111f;
            --panel: rgba(10, 18, 34, 0.78);
            --panel-2: rgba(16, 26, 48, 0.92);
            --text: #f6f8fc;
            --muted: #9daecc;
            /* premium palette: deep purple + electric teal + soft pink accent */
            --accent: #8e72ff; /* primary purple */
            --accent-2: #33e0c3; /* electric teal */
            --accent-3: #ffd07a; /* warm gold */
            --border: rgba(163, 184, 255, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 6%, rgba(124, 92, 255, 0.22), transparent 30%),
                radial-gradient(circle at 85% 14%, rgba(33, 212, 191, 0.18), transparent 26%),
                radial-gradient(circle at 60% 100%, rgba(255, 191, 105, 0.12), transparent 28%),
                linear-gradient(180deg, #07121f 0%, #0b1529 46%, #060b15 100%);
            color: var(--text);
            font-family: 'Manrope', sans-serif;
            font-size: 16px;
        }
        .block-container {
            /* increase top padding so content isn't clipped by the Streamlit header/banner */
            padding-top: 3.6rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        /* show and style the left sidebar for project navigation */
        [data-testid="stSidebar"] {
            display: block !important;
            background: linear-gradient(135deg, 
                rgba(10, 15, 35, 0.95),
                rgba(20, 15, 40, 0.92),
                rgba(15, 25, 45, 0.94));
            border-right: 1px solid rgba(142,114,255,0.08);
            padding: 1rem 0.9rem;
            box-shadow: 
                inset -8px 0 24px rgba(51,224,195,0.02),
                inset 0 0 1px rgba(142,114,255,0.04);
            width: 290px;
        }
        /* sidebar headings and text - make them vibrant */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
            color: #e8efff !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] .css-1oaqwxj {
            color: #e8efff !important;
        }
        /* sidebar label styling - use premium palette */
        [data-testid="stSidebar"] label {
            color: #d4e3ff !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }
        [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stSelectbox label {
            color: #c9deff !important;
        }
        /* keep the collapsed control visible so users can toggle the sidebar */
        [data-testid="collapsedControl"] {
            display: block !important;
        }
            /* premium primary buttons */
            button[kind="primary"] {
                background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
                color: #07111f !important;
                border-radius: 12px !important;
                padding: 0.6rem 1rem !important;
                box-shadow: 0 10px 30px rgba(33, 212, 191, 0.06) !important;
            }
            .soft-panel { border-radius: 14px; background: linear-gradient(180deg, rgba(9,16,28,0.6), rgba(20,30,48,0.45)); border:1px solid rgba(127,92,255,0.04); padding:0.9rem; }
        /* premium sidebar radio styling */
            .stSidebar [role="radiogroup"] label {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.6rem 0.6rem;
            border-radius: 12px;
            transition: background 160ms cubic-bezier(.2,.9,.2,1), transform 160ms ease, box-shadow 160ms ease;
            color: var(--text);
        }
        .stSidebar [role="radiogroup"] label:hover {
            background: linear-gradient(90deg, rgba(142,114,255,0.08), rgba(51,224,195,0.06));
            transform: translateX(6px);
            box-shadow: 0 10px 30px rgba(51,224,195,0.04);
        }
        .stSidebar .css-1d391kg { opacity: 0.98; }
        /* style native radio/checkbox accents to match the premium palette */
        .stSidebar input[type="radio"], .stSidebar input[type="checkbox"] {
            accent-color: var(--accent);
            width: 1.15rem;
            height: 1.15rem;
        }
        /* make the selected radio label more prominent (best-effort across Streamlit markup) */
        .stSidebar [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(142,114,255,0.12), rgba(51,224,195,0.08));
            box-shadow: 0 14px 40px rgba(142,114,255,0.06);
            transform: translateX(0);
        }
        .quick-guide { border-radius: 12px; padding: 0.8rem; background: linear-gradient(180deg, rgba(10,18,34,0.6), rgba(16,26,48,0.6)); border: 1px solid rgba(127,92,255,0.06); }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.02em;
        }
        p, div, span, label {
            font-family: 'Manrope', sans-serif;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.4rem 1.45rem;
            border: 1px solid var(--border);
            border-radius: 28px;
            background:
                radial-gradient(circle at 85% 15%, rgba(33, 212, 191, 0.12), transparent 18%),
                radial-gradient(circle at 10% 0%, rgba(127, 92, 255, 0.2), transparent 30%),
                linear-gradient(135deg, rgba(14, 24, 44, 0.96), rgba(8, 14, 29, 0.9));
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.3);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.45rem;
            letter-spacing: -0.04em;
            line-height: 1.02;
        }
        .hero p {
            margin: 0.35rem 0 0;
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 72ch;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }
        .hero-chip {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            background: rgba(8, 15, 30, 0.55);
            backdrop-filter: blur(10px);
        }
        .hero-chip .label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }
        .hero-chip .value {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text);
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem;
            background: linear-gradient(180deg, rgba(19, 29, 51, 0.96), rgba(10, 17, 32, 0.94));
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.2);
            min-height: 112px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--text);
        }
        .feature-card {
            border: 1px solid var(--border);
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(18, 27, 46, 0.96), rgba(8, 14, 29, 0.92));
            padding: 1rem 1rem 1rem;
            height: 100%;
            box-shadow: 0 16px 28px rgba(0, 0, 0, 0.18);
        }
        .feature-kicker {
            display: inline-block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--accent-2);
            margin-bottom: 0.45rem;
            font-weight: 800;
        }
        .feature-card h4 {
            margin: 0.1rem 0 0.3rem;
            font-size: 1.05rem;
        }
        .feature-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }
        .landing-section {
            margin-top: 1rem;
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .landing-section .rule {
            height: 1px;
            background: linear-gradient(90deg, rgba(33, 212, 191, 0.85), transparent);
            flex: 1;
        }
        .landing-section .text {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: var(--muted);
            white-space: nowrap;
        }
        .soft-panel, .stExpander {
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            background: rgba(10, 18, 34, 0.78) !important;
        }
        .stExpander details {
            border-radius: 18px !important;
        }
        .stButton > button {
            width: auto;
            border: none;
            border-radius: 14px;
            padding: 0.75rem 1rem;
            font-weight: 700;
            color: white;
            background: linear-gradient(135deg, #7c5cff 0%, #2dd4bf 100%);
            box-shadow: 0 10px 28px rgba(124, 92, 255, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 35px rgba(45, 212, 191, 0.25);
        }
        .stButton > button:focus-visible {
            outline: 2px solid rgba(33, 212, 191, 0.8);
            outline-offset: 2px;
        }
        .ghost-button > button {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: none;
        }
        .ghost-button > button:hover {
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }
        .theory {
            border-left: 3px solid var(--accent-2);
            padding: 0.75rem 1rem;
            background: rgba(11, 20, 37, 0.8);
            border-radius: 12px;
            color: #dbe7ff;
        }
        .highlight-positive {
            color: #68f0b0;
            font-weight: 700;
        }
        .highlight-negative {
            color: #ff8f9d;
            font-weight: 700;
        }
        .highlight-neutral {
            color: #93a8cb;
            font-weight: 700;
        }
        .token-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.5rem;
        }
        .token-chip {
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            font-size: 0.82rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(12, 18, 33, 0.7);
        }
        .token-pos { color: #68f0b0; }
        .token-neg { color: #ff8f9d; }
        .token-neu { color: #93a8cb; }
        .caption {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        .stSelectbox, .stSlider, .stNumberInput, .stTextInput, .stTextArea, .stFileUploader {
            margin-bottom: 0.7rem;
        }
        [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
            margin-bottom: 0.2rem;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        .stPlotlyChart {
            border: 1px solid rgba(163, 184, 255, 0.14);
            border-radius: 16px;
            padding: 0.35rem;
            background: rgba(9, 16, 31, 0.75);
        }
        @media (max-width: 1080px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        .stRadio > div, .stCheckbox {
            color: var(--text);
        }
        .stMarkdown, label, .st-b7, .st-c3, .st-d1 {
            color: var(--text);
        }
        .section-title {
            margin: 0.15rem 0 0.25rem;
            font-size: 2rem;
            letter-spacing: -0.03em;
            line-height: 1.08;
            /* ensure anchor jumps and fixed headers don't hide titles */
            scroll-margin-top: 4.2rem;
            padding-top: 0.25rem;
        }
        .section-subtitle {
            color: var(--muted);
            margin: 0 0 0.8rem;
            max-width: 70ch;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(163, 184, 255, 0.25);
            background: rgba(8, 15, 28, 0.65);
            color: #dce7ff;
            font-size: 0.92rem;
            margin-bottom: 0.65rem;
        }
        .display-title {
            font-size: clamp(2.4rem, 5vw, 5rem);
            line-height: 0.98;
            letter-spacing: -0.04em;
            margin: 0;
            font-weight: 800;
            text-align: center;
        }
        .grad-word {
            background: linear-gradient(90deg, #73b8ff, #8e72ff, #dd67ad);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .hero-copy {
            margin: 0.85rem auto 0;
            max-width: 820px;
            color: #b8c5dd;
            text-align: center;
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .why-title {
            margin: 0;
            text-align: center;
            font-size: clamp(1.85rem, 3.6vw, 3rem);
        }
        .why-subtitle {
            margin: 0.5rem 0 0.4rem;
            text-align: center;
            color: #94a5c3;
            font-size: 1rem;
        }
        .topic-card {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(14, 24, 45, 0.95), rgba(9, 16, 31, 0.93));
            padding: 1rem 1.05rem;
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        }
        .topic-pill {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            background: rgba(100, 160, 255, 0.2);
            color: #8ec9ff;
            font-size: 0.84rem;
            margin-bottom: 0.7rem;
        }
        .topic-title {
            margin: 0;
            font-size: 1.9rem;
        }
        .project-banner {
            margin: 0 0 0.75rem;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(163, 184, 255, 0.25);
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(10, 20, 38, 0.9), rgba(23, 34, 58, 0.88));
            font-family: 'Sora', sans-serif;
            font-size: clamp(1.6rem, 3.6vw, 2.6rem);
            font-weight: 800;
            letter-spacing: 0.02em;
            color: #e8efff;
            text-align: center;
        }

        /* style the Streamlit textarea to look premium */
        .stTextArea textarea {
            background: linear-gradient(180deg, rgba(12,18,34,0.7), rgba(8,14,29,0.6));
            border: 1px solid rgba(163,184,255,0.08) !important;
            color: var(--text) !important;
            padding: 1rem !important;
            border-radius: 14px !important;
            font-size: 1.02rem !important;
            min-height: 150px !important;
            box-shadow: 0 12px 30px rgba(9,16,31,0.55) inset;
        }
        .stTextArea label {
            color: var(--text) !important;
            font-weight: 600;
        }
        .stTextArea textarea::placeholder {
            color: rgba(214,224,244,0.42) !important;
        }

        /* Premium form control styling: sliders, selects, number boxes, inputs */
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 12px;
            background: linear-gradient(90deg, rgba(255,111,111,0.18), rgba(102,242,206,0.12));
            border-radius: 999px;
            outline: none;
            margin: 0.6rem 0 0.6rem 0;
        }
        input[type="range"]::-webkit-slider-runnable-track {
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255,111,111,0.18), rgba(102,242,206,0.12));
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            margin-top: -4px;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: linear-gradient(135deg, #7c5cff, #2dd4bf);
            box-shadow: 0 8px 20px rgba(45,212,191,0.18), 0 2px 6px rgba(0,0,0,0.4);
            border: 2px solid rgba(255,255,255,0.08);
        }
        input[type="range"]::-moz-range-thumb {
            width: 22px; height: 22px; border-radius: 50%;
            background: linear-gradient(135deg, #7c5cff, #2dd4bf);
            box-shadow: 0 8px 20px rgba(45,212,191,0.18), 0 2px 6px rgba(0,0,0,0.4);
            border: 2px solid rgba(255,255,255,0.08);
        }

        /* Number input & text input / select styling */
        .stNumberInput input, .stTextInput input, .stSelectbox select {
            background: linear-gradient(180deg, rgba(12,18,34,0.7), rgba(8,14,29,0.6));
            border: 1px solid rgba(127,92,255,0.08) !important;
            color: var(--text) !important;
            padding: 0.6rem 0.85rem !important;
            border-radius: 12px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 20px rgba(3,8,18,0.5) inset;
        }
        .stSelectbox .css-1gtu0r7, .stSelectbox .css-1u0qu7x {
            color: var(--text) !important;
        }
        .stNumberInput label, .stTextInput label, .stSelectbox label {
            color: #cfe6ff;
            font-weight: 700;
            letter-spacing: 0.01em;
            margin-bottom: 0.25rem;
        }

        /* slider value badges and labels - premium colors */
        .stSlider .css-1kyxreq, .stSlider label {
            color: #ffc9e3 !important;
            font-weight: 700 !important;
        }
        .stSlider div[role="slider"] {
            color: #d4e3ff !important;
        }
        /* all form labels across the app get premium treatment */
        label {
            color: #d4e3ff !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }

        /* Make the metric / value boxes more premium */
        .metric-card {
            border-radius: 18px;
            padding: 1.1rem 1rem 1rem;
            background: linear-gradient(180deg, rgba(10,18,34,0.9), rgba(18,28,48,0.85));
            border: 1px solid rgba(124,92,255,0.06);
            box-shadow: 0 18px 50px rgba(11,22,40,0.45);
        }

        /* tweak buttons to be more pill-like and colorful */
        .stButton > button {
            padding: 0.8rem 1.1rem !important;
            border-radius: 999px !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em !important;
        }

        /* subtle hover on controls */
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus, .stSelectbox select:focus {
            outline: 2px solid rgba(124,92,255,0.18) !important;
            box-shadow: 0 8px 30px rgba(124,92,255,0.06) inset !important;
        }

        /* compact project header inside sidebar (radio area already styled earlier) */
        .stSidebar .css-1d391kg, .stSidebar .stMarkdown {
            color: var(--text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


@st.cache_data(show_spinner=False)
def perceptron_data(seed: int = 7):
    X, y = make_classification(
        n_samples=180,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.25,
        flip_y=0.03,
        random_state=seed,
    )
    return X, y


@st.cache_data(show_spinner=False)
def moon_data(kind: str = "moons", seed: int = 11):
    if kind == "moons":
        return make_moons(n_samples=280, noise=0.22, random_state=seed)
    return make_circles(n_samples=280, noise=0.12, factor=0.45, random_state=seed)


@st.cache_data(show_spinner=False)
def sentiment_corpus():
    positive = [
        "I love this product and it works beautifully",
        "This is an excellent experience with great results",
        "Absolutely wonderful and surprisingly helpful",
        "The interface is clean, fast, and delightful",
        "A fantastic result with impressive quality",
        "I feel happy and confident about this",
        "The model performed brilliantly and smoothly",
        "Amazing work with a positive outcome",
        "This was a pleasant and satisfying experience",
        "The design is elegant and really useful",
        "Great job, I love the features and quality",
        "Excellent service and very professional",
        "This is awesome, highly recommended",
        "The best decision I made, very satisfied",
        "Perfect work, absolutely delightful",
        "I am impressed with this product quality",
        "Wonderful experience, will use again",
        "Fantastic features and excellent performance",
        "The quality is outstanding and impressive",
        "I really enjoyed using this application",
        "This tool is very helpful and well designed",
        "Absolutely love the simplicity and power",
        "Great tool that works perfectly",
        "This made my work so much easier and faster",
        "Fantastic user experience, very intuitive",
        "I recommend this to everyone, truly excellent",
        "Best product ever, superb quality",
        "The performance is incredible and smooth",
        "I am very happy with my purchase",
        "Excellent value and outstanding service",
        "This is the best solution available",
        "Love the attention to detail and quality",
        "Very impressive work, bravo",
        "This tool is a game changer, fantastic",
        "Absolutely thrilled with the results",
        "Great experience from start to finish",
        "I cannot praise this enough, simply amazing",
        "Perfect for my needs, excellent product",
        "Outstanding quality and great support",
        "This is the best investment I made",
    ]
    negative = [
        "I hate this product and it feels terrible",
        "This is a horrible experience with bad results",
        "Absolutely frustrating and surprisingly disappointing",
        "The interface is messy, slow, and broken",
        "A dreadful result with poor quality",
        "I feel unhappy and worried about this",
        "The model performed badly and noisily",
        "Awful work with a negative outcome",
        "This was an unpleasant and unsatisfying experience",
        "The design is clumsy and not useful",
        "Terrible experience, very disappointed",
        "This is a waste of money and time",
        "Poor quality and bad customer service",
        "I regret buying this, very bad",
        "Awful design, difficult to use",
        "This is broken and does not work",
        "Frustrating experience, not recommended",
        "Very disappointed with the results",
        "The worst product I ever tried",
        "Terrible performance and crashes often",
        "I hate how slow and buggy this is",
        "Not worth the price, very poor quality",
        "Horrible interface and confusing features",
        "This is a complete disaster and failure",
        "Bad experience, do not recommend",
        "This tool is useless and poorly designed",
        "The quality is unacceptable and bad",
        "I am very unsatisfied with this purchase",
        "Poor performance and many issues",
        "This is the worst tool I have used",
        "Dreadful experience from beginning to end",
        "I want my money back, very upset",
        "Disappointing and frustrating to use",
        "The worst investment I ever made",
        "Terrible support and bad communication",
        "This product is broken and unusable",
        "I hate everything about this tool",
        "Very bad quality and poor results",
        "Not what I expected, very disappointed",
        "This is absolutely terrible and awful",
    ]
    # Add more direct sentiment variants (insults, praise) to improve real-world handling
    positive += [
        "You are amazing",
        "I really appreciate you",
        "You did a fantastic job",
        "I love how helpful you are",
        "That's brilliant work",
        "You're so talented and kind",
        "This is exactly what I needed, thank you",
        "Really great effort and results",
    ]
    negative += [
        "You are so ugly",
        "I hate you",
        "You're the worst",
        "This is trash and useless",
        "Go away, this is awful",
        "Terrible job, very disappointing",
        "This made me angry and upset",
        "I want my money back, you broke it",
        "This is a disaster",
    ]
    neutral = [
        "It's okay, nothing special",
        "Not bad but not amazing either",
        "I have mixed feelings about this",
        "It's average and fine for now",
        "The experience was acceptable",
        "Neither good nor terrible",
        "It works but could be improved",
        "I'm indifferent about this product",
        "It's just okay, nothing to praise",
        "Mediocre but usable",
        "Neutral experience overall",
        "No strong feelings either way",
        "Works as expected, nothing more",
        "It's fine for basic use",
        "Neither impressed nor disappointed",
    ]

    texts = positive + neutral + negative
    # labels: 2 => positive, 1 => neutral, 0 => negative
    labels = np.array([2] * len(positive) + [1] * len(neutral) + [0] * len(negative))
    return texts, labels


@st.cache_resource(show_spinner=False)
def sentiment_model():
    texts, labels = sentiment_corpus()
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    lowercase=True,
                    max_features=200,
                    min_df=1,
                    max_df=0.9,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1500, random_state=42, C=1.0),
            ),
        ]
    )
    model.fit(texts, labels)
    return model


@st.cache_resource(show_spinner=False)
def digits_model():
    digits: Any = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target
    # scale features and train a small MLP on PCA-reduced features for better generalization
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=40, random_state=42)
    Xp = pca.fit_transform(Xs)

    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.2, random_state=42, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(120,), activation='relu', solver='adam', max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    # return a small wrapper pipeline to preprocess new images consistently
    class WrappedModel:
        def __init__(self, scaler, pca, clf):
            self.scaler = scaler
            self.pca = pca
            self.clf = clf

        def predict(self, Xraw):
            Xs = self.scaler.transform(Xraw)
            Xp = self.pca.transform(Xs)
            return self.clf.predict(Xp)

        def predict_proba(self, Xraw):
            Xs = self.scaler.transform(Xraw)
            Xp = self.pca.transform(Xs)
            return self.clf.predict_proba(Xp)

    wrapped = WrappedModel(scaler, pca, clf)
    return wrapped, digits.target_names, X_test, y_test


@st.cache_resource(show_spinner=False)
def lstm_like_model(seq_len: int, units: int, epochs: int):
    t = np.linspace(0, 60, 900)
    series = np.sin(t) + 0.15 * np.sin(3.2 * t) + 0.05 * np.cos(9 * t)
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, shuffle=False)
    model = MLPRegressor(
        hidden_layer_sizes=(max(12, units),),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )
    losses = []
    for _ in range(epochs):
        model.fit(X_train, y_train)
        losses.append(float(model.loss_))
    pred = model.predict(X_test)
    return model, X_test, y_test, pred, losses, series


@st.cache_resource(show_spinner=False)
def image_model():
    # Prefer a small TensorFlow CNN trained on the built-in digits when TF is available.
    # Fall back to the existing sklearn PCA+MLP pipeline if TF isn't present.
    from sklearn.datasets import load_digits

    digits: Any = load_digits()
    wrapped, target_names, X_test, y_test = digits_model()

    if HAS_TF and HAS_PIL and keras is not None and layers is not None:
        # Build image array dataset (upsample 8x8 digits to 28x28)
        imgs = []
        for im in digits.images:
            pil = Image.fromarray(np.uint8(255 - (im / im.max()) * 255)).convert("L")
            pil = pil.resize((28, 28), Image.Resampling.LANCZOS)
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            imgs.append(arr)
        X = np.stack(imgs)[..., None]
        y = digits.target.astype(np.int64)

        X_train, X_test_tf, y_train, y_test_tf = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)

        num_classes = len(np.unique(y))

        # small CNN that's quick to train on the tiny digits dataset
        assert tf is not None and keras is not None
        keras.backend.clear_session()
        model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, 3, activation="relu", padding="same"),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, activation="relu", padding="same"),
                layers.MaxPooling2D(2),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # train for a small number of epochs; cached so this runs once per session
        model.fit(X_train, y_train, validation_split=0.12, epochs=12, batch_size=32, verbose=0)

        class TFWrappedModel:
            is_tf = True

            def __init__(self, model):
                self.model = model

            def _ensure_input(self, Xraw: Any) -> np.ndarray:
                arr = np.asarray(Xraw)
                # flattened vector of 64 -> upsample to 28x28
                if arr.ndim == 1 and arr.size == 64:
                    ar = arr.reshape(8, 8)
                    pil = Image.fromarray(np.uint8(255 - (ar / (ar.max() if ar.max() > 0 else 1)) * 255)).convert("L")
                    pil = pil.resize((28, 28), Image.Resampling.LANCZOS)
                    a = np.asarray(pil, dtype=np.float32) / 255.0
                    return np.expand_dims(a, axis=(0, -1))
                # already in 28x28 float form
                if arr.ndim == 3 and arr.shape[1:] == (28, 28):
                    return arr[..., None].astype(np.float32)
                if arr.ndim == 3 and arr.shape[1:] == (28, 28, 1):
                    return arr.astype(np.float32)
                if arr.ndim == 4:
                    return arr.astype(np.float32)
                # fallback: convert PIL-like image
                try:
                    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                    a = np.asarray(img, dtype=np.float32) / 255.0
                    return np.expand_dims(a, axis=(0, -1))
                except Exception:
                    raise ValueError("Unsupported input shape for TF model")

            def predict(self, Xraw: Any):
                inp = self._ensure_input(Xraw)
                p = self.model.predict(inp, verbose=0)
                return p.argmax(axis=1)

            def predict_proba(self, Xraw: Any):
                inp = self._ensure_input(Xraw)
                return self.model.predict(inp, verbose=0)

        tfwrap = TFWrappedModel(model)

        # convert a small test set to flattened style to keep compatibility with UI code
        X_test_flat = (X_test_tf.reshape((X_test_tf.shape[0], -1)) * 255.0).astype(np.float32)
        return tfwrap, target_names, X_test_flat, y_test_tf

    # fallback: sklearn wrapped pipeline
    return wrapped, target_names, X_test, y_test


def line_boundary(weights: np.ndarray, bias: float, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, 200)
    if abs(weights[1]) < 1e-8:
        y = np.full_like(x, np.nan)
    else:
        y = -(weights[0] * x + bias) / weights[1]
    return x, y


@ignore_warnings()
def train_perceptron(eta: float, epochs: int):
    X, y = perceptron_data()
    model = Perceptron(eta0=eta, random_state=42, tol=None, fit_intercept=True)
    classes = np.array([0, 1], dtype=int)
    losses = []
    accs = []
    boundaries = []
    miscls = []
    rng = np.random.default_rng(42)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        model.partial_fit(X[order], y[order], classes=classes)
        preds = model.predict(X)
        accs.append(accuracy_score(y, preds))
        losses.append(float(np.mean(preds != y)))
        miscls.append(int(np.sum(preds != y)))
        weights = np.array(model.coef_[0], dtype=float)
        bias = float(model.intercept_[0])
        boundaries.append((weights, bias))
    return X, y, model, np.asarray(losses), np.asarray(accs), boundaries, np.asarray(miscls)


@ignore_warnings()
def train_mlp(hidden_layers: tuple[int, ...], activation: str, epochs: int, dataset_choice: str):
    X, y = moon_data(dataset_choice)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,  # type: ignore[arg-type]
        solver="adam",
        learning_rate_init=0.01,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )
    losses = []
    accs = []
    boundary_probs = []
    classes = np.array([0, 1])
    xx, yy, grid, _ = make_mesh(X)
    for _ in range(epochs):
        model.partial_fit(X, y, classes=classes)
        preds = model.predict(X)
        losses.append(float(getattr(model, "loss_", np.mean(preds != y))))
        accs.append(accuracy_score(y, preds))
        boundary_probs.append(model.predict_proba(grid)[:, 1].reshape(xx.shape))
    return X, y, model, np.asarray(losses), np.asarray(accs), xx, yy, boundary_probs


def make_mesh(X: np.ndarray, pad: float = 0.8):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid, (x_min, x_max, y_min, y_max)


def plot_binary_boundary(
    X: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    prob: np.ndarray,
    title: str,
    step_label: str,
    bg: str = "RdBu",
):
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=prob,
            colorscale=bg,
            opacity=0.55,
            contours=dict(showlines=False),
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(color=y, colorscale=[[0, "#ff6b88"], [1, "#58d6a3"]], size=8, line=dict(width=0.6, color="white")),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"{title} ({step_label})",
        height=460,
        margin=dict(l=10, r=10, t=46, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def plot_perceptron_snapshot(
    X: np.ndarray,
    y: np.ndarray,
    boundaries: list[tuple[np.ndarray, float]],
    epoch_index: int,
    title: str,
):
    xx, yy, grid, _ = make_mesh(X)
    weights, bias = boundaries[epoch_index]
    line_x, line_y = line_boundary(weights, bias, xx.min(), xx.max())
    fig = go.Figure(
        data=[
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=np.clip((grid @ weights + bias), -3, 3).reshape(xx.shape),
                colorscale="RdBu",
                opacity=0.25,
                contours=dict(showlines=False),
                showscale=False,
            ),
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(color=y, colorscale=[[0, "#ff6b88"], [1, "#58d6a3"]], size=8, line=dict(width=0.6, color="white")),
                showlegend=False,
            ),
            go.Scatter(x=line_x, y=line_y, mode="lines", line=dict(color="#66f2ce", width=3), showlegend=False),
        ],
    )
    fig.update_layout(
        title=f"{title} (Epoch {epoch_index + 1})",
        height=460,
        margin=dict(l=10, r=10, t=46, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def highlighted_sentence(text: str, model: Pipeline) -> str:
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    analyzer = tfidf.build_analyzer()
    tokens = analyzer(text)
    if not tokens:
        return html.escape(text)
    vocab = tfidf.vocabulary_
    coefs = clf.coef_
    classes = list(clf.classes_)
    # map class index to label
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    spans = []
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            # for multiclass, pick the class with the largest coefficient for this token
            token_coefs = coefs[:, idx]
            best_class = int(np.argmax(token_coefs))
            if label_map.get(best_class, "Neutral") == "Positive":
                klass = "highlight-positive"
            elif label_map.get(best_class) == "Negative":
                klass = "highlight-negative"
            else:
                klass = "highlight-neutral"
            spans.append(f'<span class="{klass}">{html.escape(token)}</span>')
        else:
            spans.append(html.escape(token))
    return " ".join(spans)


def sentiment_focus_tokens(text: str, model: Pipeline, top_k: int = 8) -> list[tuple[str, str, float]]:
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    analyzer = tfidf.build_analyzer()
    tokens = analyzer(text)
    if not tokens:
        return []
    vocab = tfidf.vocabulary_
    coefs = clf.coef_
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    scored = []
    seen = set()
    for token in tokens:
        if token in vocab and token not in seen:
            idx = vocab[token]
            token_coefs = coefs[:, idx]
            best_class = int(np.argmax(np.abs(token_coefs)))
            label = label_map.get(best_class, "Neutral")
            score = float(token_coefs[best_class])
            scored.append((token, label, abs(score)))
            seen.add(token)
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def image_to_digit_features(img: Image.Image) -> np.ndarray:
    # Robust preprocessing pipeline:
    # 1) convert to grayscale
    # 2) threshold and crop to bounding box of the digit
    # 3) paste centered onto a square canvas, resize to 20x20, then to 8x8
    gray = ImageOps.grayscale(img)
    arr_full = np.asarray(gray, dtype=np.uint8)

    # adaptive threshold: compute Otsu-like threshold from image histogram
    try:
        thresh = np.mean(arr_full) * 0.85
    except Exception:
        thresh = 128
    bw = (arr_full < thresh).astype(np.uint8)  # foreground as 1

    # find bounding box of foreground
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        # empty image, fallback to center-crop
        cropped = gray
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = gray.crop((x0, y0, x1 + 1, y1 + 1))

    # paste onto square canvas and center
    max_side = max(cropped.size)
    square = Image.new("L", (max_side, max_side), color=255)
    paste_x = (max_side - cropped.size[0]) // 2
    paste_y = (max_side - cropped.size[1]) // 2
    square.paste(cropped, (paste_x, paste_y))

    # resize to 20x20 first (preserving stroke), then to 8x8
    small = square.resize((20, 20), Image.Resampling.LANCZOS)
    final = small.resize((8, 8), Image.Resampling.LANCZOS)

    arr = np.asarray(final, dtype=np.float32)
    # invert so that foreground is large values similar to sklearn digits
    arr = 255.0 - arr

    # normalize to 0-16
    arr = (arr - arr.min())
    if arr.max() > 0:
        arr = arr / arr.max() * 16.0

    return arr.ravel()


def image_to_cnn_array(img: Image.Image) -> np.ndarray:
    """Prepare a 28x28 grayscale float32 array for a CNN (shape: 1,28,28,1)."""
    gray = ImageOps.grayscale(img)
    # center on square canvas
    max_side = max(gray.size)
    square = Image.new("L", (max_side, max_side), color=255)
    paste_x = (max_side - gray.size[0]) // 2
    paste_y = (max_side - gray.size[1]) // 2
    square.paste(gray, (paste_x, paste_y))
    img28 = square.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(img28, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=(0, -1))


def show_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_view(fig: go.Figure) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
        },
    )


def sidebar_nav() -> str:
    st.sidebar.markdown(
        """
        <div style="border:1px solid rgba(163,184,255,0.18);border-radius:16px;padding:0.9rem 0.9rem 0.8rem;background:linear-gradient(145deg, rgba(18,29,52,0.95), rgba(9,16,31,0.95));margin-bottom:0.65rem;">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.15em;color:#92a7ca;">DL Lab</div>
            <div style="font-size:1.25rem;font-weight:800;color:#f6f8fc;line-height:1.1;margin-top:0.2rem;">Deep Learning Toolbox</div>
            <div style="font-size:0.88rem;color:#9daecc;margin-top:0.35rem;">Interactive simulations for neural model intuition.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    choice = st.sidebar.radio(
        "Navigate Modules",
        ["Home", "Perceptron", "Multi-Layer Perceptron (MLP)", "Sentiment Analysis", "Computer Vision", "Hopfield Network", "LSTM"],
        key="nav_choice",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="quick-guide">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#93a8cb;">Quick guide</div>
            <div style="font-size:0.9rem;color:#d7e4ff;line-height:1.5;margin-top:0.25rem;">1) Pick a module<br>2) Tune controls<br>3) Move simulation step<br>4) Read curves and boundary together</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return choice


def set_section(section: str) -> None:
    st.session_state.nav_choice = section


def section_header(title: str, subtitle: str) -> None:
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def landing_chip(label: str, value: str) -> str:
    return f"<div class='hero-chip'><div class='label'>{label}</div><div class='value'>{value}</div></div>"


def feature_card(kicker: str, title: str, body: str) -> str:
    return f"""
    <div class='feature-card'>
        <div class='feature-kicker'>{kicker}</div>
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    """


def topic_card(kicker: str, title: str) -> str:
    return f"""
    <div class='topic-card'>
        <div class='topic-pill'>{kicker}</div>
        <h4>{title}</h4>
    </div>
    """


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div style="text-align:center;">
                <div class="hero-badge">✦ Interactive Deep Learning Platform</div>
            </div>
            <h1 class="display-title">Master <span class="grad-word">Deep Learning</span> Visually</h1>
            <p class="hero-copy">
                Experience neural networks through interactive visualizations and hands-on simulation.
                From perceptrons to sequence models, understand complex ideas quickly by seeing model
                behavior update in real time.
            </p>
            <div class="hero-grid">
                <div class="hero-chip"><div class="label">Tagline</div><div class="value">Learn by seeing, not memorizing</div></div>
                <div class="hero-chip"><div class="label">Built for</div><div class="value">Students, interview prep, self-study</div></div>
                <div class="hero-chip"><div class="label">Coverage</div><div class="value">Perceptron, MLP, CV, NLP, Hopfield, LSTM</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='landing-section'><span class='text'>Get started</span><div class='rule'></div></div>", unsafe_allow_html=True)
    launch_col1, launch_col2, launch_col3 = st.columns(3, gap="medium")
    with launch_col1:
        st.button("Start with Perceptron", on_click=set_section, args=("Perceptron",))
    with launch_col2:
        st.button("Try MLP Lab", on_click=set_section, args=("Multi-Layer Perceptron (MLP)",))
    with launch_col3:
        st.button("Explore LSTM", on_click=set_section, args=("LSTM",))

    stat1, stat2, stat3, stat4 = st.columns(4, gap="medium")
    with stat1:
        show_metric("Learning modes", "6")
    with stat2:
        show_metric("Visual outputs", "Boundaries + charts")
    with stat3:
        show_metric("Text + vision", "Sentiment + image demos")
    with stat4:
        show_metric("Sequence models", "LSTM-style prediction")

    st.markdown("<div class='landing-section'><span class='text'>Why use it</span><div class='rule'></div></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='why-title'>Why <span class='grad-word'>DeepLearn</span>?</h2>", unsafe_allow_html=True)
    st.markdown("<p class='why-subtitle'>Premium features designed for modern learners</p>", unsafe_allow_html=True)

    why1, why2, why3, why4 = st.columns(4, gap="medium")
    with why1:
        st.markdown(feature_card("Live visuals", "Real-Time Visualizations", "Watch boundaries, losses, and probability maps update while you tune parameters."), unsafe_allow_html=True)
    with why2:
        st.markdown(feature_card("Hands-on", "Learn by Experimenting", "Adjust controls and instantly observe how architectural changes alter behavior."), unsafe_allow_html=True)
    with why3:
        st.markdown(feature_card("Balanced", "Theory + Practice", "Short conceptual blocks are paired with practical, interactive model demos."), unsafe_allow_html=True)
    with why4:
        st.markdown(feature_card("Structured", "Progressive Learning", "Move from fundamentals to advanced modules like CNN-style vision and sequence learning."), unsafe_allow_html=True)

    st.markdown("<div class='landing-section'><span class='text'>Project modules</span><div class='rule'></div></div>", unsafe_allow_html=True)
    st.markdown("<p class='why-subtitle'>Click any module to open that concept directly.</p>", unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3, gap="medium")
    with t1:
        st.markdown(topic_card("Module 1", "Perceptron"), unsafe_allow_html=True)
        st.button("Open Perceptron", key="topic_perceptron", on_click=set_section, args=("Perceptron",))
    with t2:
        st.markdown(topic_card("Module 2", "Multi-Layer Perceptron (MLP)"), unsafe_allow_html=True)
        st.button("Open MLP", key="topic_mlp", on_click=set_section, args=("Multi-Layer Perceptron (MLP)",))
    with t3:
        st.markdown(topic_card("Module 3", "Sentiment Analysis"), unsafe_allow_html=True)
        st.button("Open Sentiment", key="topic_sentiment", on_click=set_section, args=("Sentiment Analysis",))

    t4, t5, t6 = st.columns(3, gap="medium")
    with t4:
        st.markdown(topic_card("Module 4", "Computer Vision"), unsafe_allow_html=True)
        st.button("Open Vision", key="topic_vision", on_click=set_section, args=("Computer Vision",))
    with t5:
        st.markdown(topic_card("Module 5", "Hopfield Network"), unsafe_allow_html=True)
        st.button("Open Hopfield", key="topic_hopfield", on_click=set_section, args=("Hopfield Network",))
    with t6:
        st.markdown(topic_card("Module 6", "LSTM"), unsafe_allow_html=True)
        st.button("Open LSTM", key="topic_lstm", on_click=set_section, args=("LSTM",))

    st.markdown("<div class='landing-section'><span class='text'>How to use</span><div class='rule'></div></div>", unsafe_allow_html=True)
    guide = st.columns(3, gap="medium")
    with guide[0]:
        st.markdown(feature_card("Step 1", "Choose a module", "Pick a topic from the sidebar or use the launch buttons above to jump straight in."), unsafe_allow_html=True)
    with guide[1]:
        st.markdown(feature_card("Step 2", "Adjust controls", "Move sliders, switch datasets, and change activation functions to see the models react."), unsafe_allow_html=True)
    with guide[2]:
        st.markdown(feature_card("Step 3", "Read the visuals", "Use the plots and small theory cards to connect parameter changes to the underlying behavior."), unsafe_allow_html=True)

    # removed end-of-section expander for a cleaner layout


def render_project_banner() -> None:
    st.markdown("<div class='project-banner'><span class='grad-word'>DeepLearn</span></div>", unsafe_allow_html=True)


if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = "Home"

sidebar_nav()
render_project_banner()
choice = st.session_state.nav_choice

if choice == "Home":
    render_home()

elif choice == "Perceptron":
    section_header(
        "Perceptron",
        "See how a single-layer classifier learns a line that separates two classes. The boundary updates after each epoch so you can connect the controls to the geometry.",
    )
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.markdown("<div class='theory'><strong>Idea:</strong> a perceptron learns a linear separator by nudging a hyperplane toward misclassified points.</div>", unsafe_allow_html=True)
        eta = st.slider("Learning rate", 0.01, 1.0, 0.12, 0.01, help="Smaller values learn more carefully; larger values move faster but can overshoot.")
        epochs = st.slider("Epochs", 1, 40, 12, 1, help="Each epoch makes one pass over the data.")
        reset = st.button("Reset perceptron")
    with st.spinner("Training perceptron..."):
        if reset:
            st.cache_data.clear()
        X, y, model, losses, accs, boundaries, miscls = train_perceptron(eta, epochs)
    sim_epoch = st.slider(
        "Simulation step (epoch)",
        min_value=1,
        max_value=epochs,
        value=epochs,
        help="Move this slider to inspect how the boundary and metrics evolve during training.",
    )
    sim_idx = sim_epoch - 1
    c1, c2, c3 = st.columns(3)
    with c1:
        show_metric("Accuracy", f"{accs[sim_idx] * 100:.1f}%")
    with c2:
        show_metric("Loss", f"{losses[sim_idx]:.3f}")
    with c3:
        show_metric("Misclassified", str(int(miscls[sim_idx])))
    viz_left, viz_right = st.columns([1.35, 1], gap="large")
    with viz_left:
        perceptron_fig = plot_perceptron_snapshot(X, y, boundaries, sim_idx, "Decision boundary")
        plotly_view(perceptron_fig)
    with viz_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=accs, mode="lines+markers", line=dict(color="#66f2ce", width=3), name="Accuracy"))
        fig.add_trace(go.Scatter(y=losses, mode="lines+markers", line=dict(color="#ff8f9d", width=3), name="0/1 Loss"))
        fig.add_vline(x=sim_idx, line_width=2, line_dash="dash", line_color="#ffbf69")
        fig.update_layout(
            title=f"Training curves (Epoch {sim_epoch})",
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=8),
        )
        plotly_view(fig)
    # removed end-of-section expander for a cleaner layout

elif choice == "Multi-Layer Perceptron (MLP)":
    section_header(
        "Multi-Layer Perceptron",
        "Move from straight-line behavior to a non-linear decision surface. Hidden layers and activation functions shape the boundary you see in the chart.",
    )
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.markdown("<div class='theory'><strong>Idea:</strong> hidden layers let the network bend its decision surface to fit non-linear patterns.</div>", unsafe_allow_html=True)
        dataset_choice = st.selectbox("Dataset", ["moons", "circles"], help="Moons is slightly easier; circles emphasizes the need for non-linearity.")
        hidden_layers_n = st.slider("Hidden layers", 1, 4, 2, 1, help="How many internal layers to stack.")
        neurons = st.slider("Neurons per layer", 4, 60, 18, 2, help="More neurons increase flexibility.")
        activation = st.selectbox("Activation function", ["relu", "tanh", "logistic"], help="Choose the non-linearity for each hidden layer.")
        epochs = st.slider("Training epochs", 2, 50, 18, 1)
        reset = st.button("Reset MLP")
    with st.spinner("Training MLP..."):
        if reset:
            st.cache_data.clear()
        hidden_layers = tuple([neurons] * hidden_layers_n)
        X, y, model, losses, accs, xx, yy, boundary_probs = train_mlp(hidden_layers, activation, epochs, dataset_choice)
    sim_epoch = st.slider(
        "Simulation step (epoch)",
        min_value=1,
        max_value=epochs,
        value=epochs,
        key="mlp_step",
        help="Inspect how the non-linear boundary forms as training progresses.",
    )
    sim_idx = sim_epoch - 1
    c1, c2, c3 = st.columns(3)
    with c1:
        show_metric("Accuracy", f"{accs[sim_idx] * 100:.1f}%")
    with c2:
        show_metric("Loss", f"{losses[sim_idx]:.3f}")
    with c3:
        show_metric("Architecture", f"{hidden_layers_n} x {neurons}")
    viz_left, viz_right = st.columns([1.4, 1], gap="large")
    with viz_left:
        mlp_fig = plot_binary_boundary(
            X,
            y,
            xx,
            yy,
            boundary_probs[sim_idx],
            "Decision boundary",
            f"Epoch {sim_epoch}",
        )
        plotly_view(mlp_fig)
    with viz_right:
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=losses, mode="lines+markers", line=dict(color="#ffca63", width=3), name="Loss"))
        loss_fig.add_trace(go.Scatter(y=accs, mode="lines+markers", line=dict(color="#66f2ce", width=3), name="Accuracy", yaxis="y2"))
        loss_fig.add_vline(x=sim_idx, line_width=2, line_dash="dash", line_color="#ffbf69")
        loss_fig.update_layout(
            title=f"Training progress (Epoch {sim_epoch})",
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=8),
            yaxis=dict(title="Loss"),
            yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
        )
        plotly_view(loss_fig)
    # removed end-of-section expander for a cleaner layout

elif choice == "Sentiment Analysis":
    section_header(
        "Sentiment Analysis",
        "Type a sentence and watch the model score it as positive, neutral, or negative. The probability bars and highlighted words make the prediction easier to interpret.",
    )
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown("<div class='theory'><strong>Idea:</strong> text is converted into features, and the classifier estimates positive vs negative intent.</div>", unsafe_allow_html=True)
        text = st.text_area(
            "Enter a sentence",
            "This toolbox feels beautiful, fast, and surprisingly helpful.",
            height=140,
            help="Try words like amazing, frustrating, useful, or broken to see the score change.",
        )
        reset = st.button("Reset sentiment input")
        if reset:
            st.session_state["sentiment_text"] = ""
    model = sentiment_model()
    probs = model.predict_proba([text])[0]
    # classes are encoded as 0:Negative, 1:Neutral, 2:Positive
    class_names = ["Negative", "Neutral", "Positive"]
    best_idx = int(np.argmax(probs))
    label = class_names[best_idx]
    confidence = probs[best_idx]

    # simple rule-based postprocessing to catch direct insults and low-confidence neutral cases
    text_low = (text or "").lower()
    negative_cues = ["hate", "ugly", "worst", "trash", "disgust", "idiot", "stupid", "awful", "terrible", "dumb"]
    positive_cues = ["love", "amazing", "great", "brilliant", "fantastic", "wonderful", "best", "excellent", "beautiful", "thanks", "thank you", "appreciate"]
    if any(c in text_low for c in negative_cues):
        label = "Negative"
        confidence = max(confidence, 0.78)
    elif any(c in text_low for c in positive_cues):
        label = "Positive"
        confidence = max(confidence, 0.72)
    else:
        # low-confidence -> neutral
        if float(confidence) < 0.55:
            label = "Neutral"
            # boost displayed confidence to be the max of the two top classes
            sorted_probs = sorted(list(probs), reverse=True)
            confidence = max(confidence, min(0.55, sorted_probs[0]))
    c1, c2 = st.columns(2)
    with c1:
        show_metric("Sentiment", label)
    with c2:
        show_metric("Confidence", f"{confidence * 100:.1f}%")
    chart = go.Figure(
        data=[
            go.Bar(
                x=class_names,
                y=[probs[0], probs[1], probs[2]],
                marker_color=["#ff8f9d", "#7c88a8", "#66f2ce"],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="auto",
            )
        ]
    )
    chart.update_layout(
        title="Probability view",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(chart, width="stretch")
    focus_tokens = sentiment_focus_tokens(text, model)
    chips = []
    for token, label_name, _ in focus_tokens:
        cls = "token-neu"
        if label_name == "Positive":
            cls = "token-pos"
        elif label_name == "Negative":
            cls = "token-neg"
        chips.append(f"<span class='token-chip {cls}'>{html.escape(token)}</span>")
    chip_row = "".join(chips) if chips else "<span class='token-chip token-neu'>No keywords detected</span>"
    st.markdown(
        f"""
        <div class='soft-panel' style='padding:1rem 1.1rem;'>
            <strong>Word focus:</strong>
            <div class='token-row'>{chip_row}</div>
            <div style='margin-top:0.7rem;color:var(--muted);'>Sentence emphasis:</div>
            <div style='margin-top:0.35rem;'>{highlighted_sentence(text, model)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif choice == "Computer Vision":
    section_header(
        "Computer Vision",
        "Upload an image, inspect the predicted class, and compare lightweight transforms that mimic the way a vision pipeline exposes structure.",
    )
    if not HAS_PIL:
        st.error("⚠️ Computer Vision module requires Pillow (PIL), which is not available in this deployment environment.")
        st.info("This module works perfectly locally. To run locally: `pip install Pillow` then `streamlit run app.py`")
    else:
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.markdown("<div class='theory'><strong>Idea:</strong> a vision model turns pixels into feature maps and class probabilities.</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
            # runtime samples generated from sklearn digits so users don't need external files
            sample_digits: Any = load_digits()
            # choose representative samples by digit label to ensure thumbnails match expected digits
            preferred_digits = [2, 6, 1, 5, 3]
            sample_indices = []
            for d in preferred_digits:
                idxs = np.where(sample_digits.target == d)[0]
                if idxs.size:
                    sample_indices.append(int(idxs[0]))
            # fallback: fill with the first unique indices until we have 5
            if len(sample_indices) < 5:
                for i in range(len(sample_digits.images)):
                    if i not in sample_indices:
                        sample_indices.append(i)
                    if len(sample_indices) >= 5:
                        break
            st.markdown("<div style='margin-top:0.6rem;color:var(--muted)'>Or pick a sample:</div>", unsafe_allow_html=True)
            thumbs = st.columns(len(sample_indices))
            for col, idx in zip(thumbs, sample_indices):
                with col:
                    sample_img = sample_digits.images[idx]
                    # normalize to 0-255, invert so digit is dark, and improve contrast
                    arr255 = (sample_img / (sample_img.max() if sample_img.max() > 0 else 1.0) * 255.0).astype(np.uint8)
                    thumb = Image.fromarray(255 - arr255).convert("RGB")
                    # keep pixel shape sharp for handwritten digits
                    thumb = thumb.resize((88, 88), Image.Resampling.NEAREST)
                    thumb = ImageOps.autocontrast(thumb, cutoff=1)
                    st.image(thumb, width=88)
                    if st.button(f"Load {idx}", key=f"sample_btn_{idx}"):
                        st.session_state["cv_sample_idx"] = idx

            controls = st.columns([1, 1, 1])
            with controls[0]:
                if st.button("Shuffle sample"):
                    st.session_state["cv_sample_idx"] = int(np.random.default_rng().choice(sample_indices))
            with controls[1]:
                if st.button("Clear sample"):
                    st.session_state["cv_sample_idx"] = None
            with controls[2]:
                current = st.session_state.get("cv_sample_idx", None)
                label = f"Loaded: {current}" if current is not None else "Loaded: None"
                st.markdown(f"<div style='margin-top:0.45rem;color:var(--muted)'>{label}</div>", unsafe_allow_html=True)

            sample_choice = st.session_state.get("cv_sample_idx", None)
            reset = st.button("Reset vision demo")
            st.caption("Tip: this offline demo classifier works best with simple handwritten digits or high-contrast icons.")
        clf, target_names, X_test, y_test = image_model()
        is_tf_model = getattr(clf, "is_tf", False)

        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            if is_tf_model:
                inp = image_to_cnn_array(img)
                probs = clf.predict_proba(inp)[0]
                pred = int(np.argmax(probs))
            else:
                pred_vec = image_to_digit_features(img)
                pred = clf.predict([pred_vec])[0]
                probs = clf.predict_proba(np.asarray([pred_vec]))[0]
        elif sample_choice is not None:
            idx = int(sample_choice)
            sample_img = sample_digits.images[idx]
            img = Image.fromarray(np.uint8(255 - (sample_img / sample_img.max()) * 255)).convert("RGB")
            if is_tf_model:
                inp = image_to_cnn_array(img)
                probs = clf.predict_proba(inp)[0]
                pred = int(np.argmax(probs))
            else:
                pred_vec = sample_digits.data[idx]
                pred = clf.predict([pred_vec])[0]
                probs = clf.predict_proba(np.asarray([pred_vec]))[0]
        else:
            # default sample
            sample_img = sample_digits.images[12]
            img = Image.fromarray(np.uint8(255 - (sample_img / sample_img.max()) * 255)).convert("RGB")
            if is_tf_model:
                inp = image_to_cnn_array(img)
                probs = clf.predict_proba(inp)[0]
                pred = int(np.argmax(probs))
            else:
                pred_vec = sample_digits.data[12]
                pred = clf.predict([pred_vec])[0]
                probs = clf.predict_proba(np.asarray([pred_vec]))[0]
        if reset:
            st.session_state["cv_sample_idx"] = None
            st.cache_data.clear()
        c1, c2 = st.columns(2)
        with c1:
            show_metric("Predicted class", str(pred))
        with c2:
            show_metric("Confidence", f"{probs[pred] * 100:.1f}%")
        img_col, viz_col = st.columns([1, 1.1], gap="large")
        with img_col:
            st.image(img, caption="Uploaded image", use_container_width=True)
            gray = ImageOps.grayscale(img)
            edges = Image.fromarray(np.uint8(np.clip(ndimage.sobel(np.asarray(gray, dtype=float)), 0, 255)))
            a, b, c = st.columns(3)
            with a:
                st.image(gray, caption="Grayscale", use_container_width=True)
            with b:
                st.image(ImageOps.autocontrast(gray.filter(ImageFilter.EDGE_ENHANCE_MORE)), caption="Enhanced", use_container_width=True)
            with c:
                st.image(edges, caption="Edges", use_container_width=True)
        with viz_col:
            cv_fig = go.Figure(
                data=[
                    go.Bar(x=[str(i) for i in range(len(probs))], y=probs, marker_color="#66f2ce"),
                ]
            )
            cv_fig.update_layout(
                title="Class probabilities",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(cv_fig, use_container_width=True)
            st.markdown("<div class='soft-panel' style='padding:1rem 1.1rem;'>Feature maps are approximated here with classic image transforms so the app remains fully usable offline.</div>", unsafe_allow_html=True)
        # removed end-of-section expander for a cleaner layout

elif choice == "Hopfield Network":
    section_header(
        "Hopfield Network",
        "Flip bits, add noise, and watch an associative memory settle back toward the stored pattern.",
    )
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown("<div class='theory'><strong>Idea:</strong> a Hopfield network iteratively restores a stored binary pattern from noisy input.</div>", unsafe_allow_html=True)
        noise = st.slider("Noise level", 0.0, 0.5, 0.18, 0.01, help="How many pixels to flip before recovery starts.")
        pattern_size = st.select_slider("Grid size", options=[4, 6, 8], value=6)
        reset = st.button("Reset Hopfield")
    base_pattern = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    if pattern_size == 4:
        base_pattern = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=int)
    elif pattern_size == 8:
        base_pattern = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=int,
        )
    rng = np.random.default_rng(42)
    noisy = base_pattern.copy().reshape(-1)
    flip_count = max(1, int(noise * noisy.size))
    flip_idx = rng.choice(noisy.size, flip_count, replace=False)
    noisy[flip_idx] = 1 - noisy[flip_idx]
    noisy = noisy.reshape(base_pattern.shape)
    stored = base_pattern.reshape(-1) * 2 - 1
    weights = np.outer(stored, stored)
    np.fill_diagonal(weights, 0)
    state = noisy.reshape(-1) * 2 - 1
    states = [state.copy()]
    recovery_steps = st.slider(
        "Recovery iterations",
        min_value=2,
        max_value=12,
        value=7,
        help="How many asynchronous update rounds to simulate.",
    )
    for _ in range(recovery_steps):
        state = np.sign(weights @ state)
        state[state == 0] = 1
        states.append(state.copy())
    sim_step = st.slider(
        "Simulation step (iteration)",
        min_value=0,
        max_value=len(states) - 1,
        value=len(states) - 1,
        help="Move this slider to inspect how the pattern is corrected over iterations.",
    )
    selected_state = states[sim_step]
    selected_binary = ((selected_state.reshape(base_pattern.shape) + 1) // 2).astype(int)
    hamming = np.sum(selected_state != stored)
    hamming_by_step = [int(np.sum(s != stored)) for s in states]
    c1, c2, c3 = st.columns(3)
    with c1:
        show_metric("Flip count", str(flip_count))
    with c2:
        show_metric("Selected step", str(sim_step))
    with c3:
        show_metric("Hamming distance", str(int(hamming)))
    left_plot, right_plot = st.columns([1, 1], gap="large")
    with left_plot:
        heat = go.Figure(go.Heatmap(z=selected_binary, colorscale=[[0, "#101d34"], [1, "#67f0bf"]], showscale=False))
        heat.update_layout(
            title=f"Pattern state (Iteration {sim_step})",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=8),
        )
        plotly_view(heat)
    with right_plot:
        hopfield_curve = go.Figure()
        hopfield_curve.add_trace(
            go.Scatter(y=hamming_by_step, mode="lines+markers", line=dict(color="#ffbf69", width=3), name="Hamming")
        )
        hopfield_curve.add_vline(x=sim_step, line_width=2, line_dash="dash", line_color="#66f2ce")
        hopfield_curve.update_layout(
            title="Recovery progress",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=8),
            yaxis_title="Mismatched bits",
            xaxis_title="Iteration",
        )
        plotly_view(hopfield_curve)
    # removed end-of-section expander for a cleaner layout

elif choice == "LSTM":
    section_header(
        "LSTM Sequence Prediction",
        "Use a noisy signal to compare the actual curve with the model forecast. This section shows how sequence models learn temporal patterns.",
    )
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("<div class='theory'><strong>Idea:</strong> a sequence model reads a window of values and predicts the next one.</div>", unsafe_allow_html=True)
        seq_len = st.slider("Sequence length", 8, 40, 18, 1, help="How many past points the model sees at once.")
        units = st.slider("LSTM units", 8, 64, 24, 2, help="Used as the width of the recurrent-style fallback model.")
        epochs = st.slider("Epochs", 5, 60, 25, 1)
        reset = st.button("Reset sequence model")
    if HAS_TF and keras is not None and layers is not None:
        with st.spinner("Training Keras LSTM..."):
            t = np.linspace(0, 60, 900)
            series = np.sin(t) + 0.15 * np.sin(3.2 * t) + 0.05 * np.cos(9 * t)
            X, y = [], []
            for i in range(len(series) - seq_len):
                X.append(series[i : i + seq_len])
                y.append(series[i + seq_len])
            X = np.asarray(X)[..., None]
            y = np.asarray(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, shuffle=False)
            if reset and tf is not None:
                tf.keras.backend.clear_session()  # type: ignore[union-attr]
            model = keras.Sequential(
                [
                    layers.Input(shape=(seq_len, 1)),
                    layers.LSTM(units, activation="tanh"),
                    layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            history = model.fit(X_train, y_train, validation_split=0.15, epochs=epochs, verbose=0)
            pred = model.predict(X_test, verbose=0).ravel()
            losses = history.history["loss"]
    else:
        with st.spinner("Training sequence model..."):
            if reset:
                st.cache_resource.clear()
            _, X_test, y_test, pred, losses, series = lstm_like_model(seq_len, units, epochs)
    point_step = st.slider(
        "Simulation step (prediction horizon)",
        min_value=max(8, min(40, len(y_test) // 10)),
        max_value=len(y_test),
        value=len(y_test),
        help="Reveal prediction quality over time by increasing the visible horizon.",
    )
    loss_step = st.slider(
        "Simulation step (training epoch)",
        min_value=1,
        max_value=len(losses),
        value=len(losses),
        help="Inspect optimization progress up to a selected training epoch.",
    )
    point_idx = point_step - 1
    point_error = abs(float(y_test[point_idx] - pred[point_idx]))
    c1, c2, c3 = st.columns(3)
    with c1:
        show_metric("Loss @ epoch", f"{losses[loss_step - 1]:.4f}")
    with c2:
        show_metric("Visible horizon", str(point_step))
    with c3:
        show_metric("Point abs error", f"{point_error:.4f}")
    left_plot, right_plot = st.columns([1.2, 1], gap="large")
    with left_plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test[:point_step], mode="lines", name="Actual", line=dict(color="#67f0bf", width=3)))
        fig.add_trace(go.Scatter(y=pred[:point_step], mode="lines", name="Predicted", line=dict(color="#ffca63", width=3, dash="dash")))
        fig.add_vline(x=point_idx, line_width=2, line_dash="dash", line_color="#66f2ce")
        fig.update_layout(
            title=f"Actual vs predicted (step {point_step})",
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=8),
        )
        plotly_view(fig)
    with right_plot:
        loss_fig = go.Figure(go.Scatter(y=losses[:loss_step], mode="lines+markers", line=dict(color="#66f2ce", width=3)))
        loss_fig.add_vline(x=loss_step - 1, line_width=2, line_dash="dash", line_color="#ffbf69")
        loss_fig.update_layout(
            title=f"Training loss (epoch {loss_step})",
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=8),
        )
        plotly_view(loss_fig)
        st.markdown("<div class='soft-panel' style='padding:1rem 1.1rem;'>This section uses TensorFlow/Keras when available, otherwise it falls back to a sequence regressor that keeps the interface responsive and educational.</div>", unsafe_allow_html=True)
    # removed end-of-section expander for a cleaner layout

else:
    st.info("Choose a section from the sidebar.")
