import streamlit as st
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="arXiv Classifier", page_icon="📚")
st.title("📚 arXiv Article Classifier")

# Демо-категории (если модель не загрузится)
DEMO_CATEGORIES = {
    0: "🤖 Artificial Intelligence",
    1: "💬 Natural Language Processing",
    2: "👁️ Computer Vision",
    3: "⚛️ Physics",
    4: "📐 Mathematics",
}

@st.cache_resource
def load_model():
    """Загрузка модели с обработкой ошибок"""
    try:
        # Пробуем загрузить с Hugging Face Hub
        model = DistilBertForSequenceClassification.from_pretrained("sofia-ol/arxiv-classifier")
        tokenizer = AutoTokenizer.from_pretrained("sofia-ol/arxiv-classifier")
        model.eval()
        st.success("✅ Модель успешно загружена!")
        return model, tokenizer, True
    except Exception as e:
        st.warning(f"⚠️ Демо-режим: модель не найдена ({str(e)[:100]}...)")
        return None, None, False

def predict_demo(text, threshold=0.95):
    """Демо-предсказания"""
    probs = np.random.dirichlet(np.ones(len(DEMO_CATEGORIES)))
    indices = np.argsort(probs)[::-1]
    cumulative = 0
    results = []
    for idx in indices:
        prob = probs[idx]
        cumulative += prob
        results.append((DEMO_CATEGORIES[idx], prob, f"class_{idx}"))
        if cumulative >= threshold:
            break
    return results

title = st.text_area("📌 Название статьи *", height=100)
abstract = st.text_area("📄 Аннотация", height=200)

if st.button("🔍 Классифицировать"):
    if not title.strip():
        st.error("Введите название")
    else:
        full_text = title
        if abstract.strip():
            full_text = title + " [SEP] " + abstract
        
        with st.spinner("Анализирую..."):
            model, tokenizer, has_model = load_model()
            
            if has_model:
                # Используйте реальную модель
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=384)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    probs = probs.numpy().squeeze()
                # ... обработка реальных результатов
                st.info("Режим: реальная модель")
            else:
                # Используем демо
                predictions = predict_demo(full_text)
                st.info("ℹ️ Демо-режим (случайные предсказания)")
                for name, prob, _ in predictions:
                    st.write(f"**{name}**")
                    st.progress(float(prob), text=f"{prob*100:.1f}%")
