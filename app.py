import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn import Softmax
import numpy as np
import gdown
import zipfile
import os

# Настройка страницы
st.set_page_config(
    page_title="arXiv Classifier",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def load_model():
    
    # Скачиваем модель
    url = f"https://drive.google.com/file/d/19y-qN-IKmw8ymKde1eShkEkQEnsAgzJV/view?usp=drive_link"
    zip_path = "/tmp/best_model.zip"
    extract_path = "/tmp/best_model"
    
    if not os.path.exists(extract_path):
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
    
    # Загружаем модель
    model = DistilBertForSequenceClassification.from_pretrained(extract_path)
    tokenizer = AutoTokenizer.from_pretrained(extract_path)
    model.eval()
    return model, tokenizer

def predict(text, model, tokenizer, threshold=0.95):
    """Предсказание с накоплением вероятностей до 95%"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = Softmax(dim=1)(outputs.logits).squeeze().numpy()
    
    # Сортировка и накопление
    indices = np.argsort(probs)[::-1]
    cumulative = 0
    results = []
    
    for idx in indices:
        prob = probs[idx]
        cumulative += prob
        category = model.config.id2label[idx]
        # Человекочитаемые названия категорий
        category_names = {
            'cs.AI': '🤖 Искусственный интеллект',
            'cs.CL': '💬 Обработка естественного языка',
            'cs.CV': '👁️ Компьютерное зрение',
            'physics': '⚛️ Физика',
            'math': '📐 Математика',
            'q-bio': '🧬 Биология'
        }
        display_name = category_names.get(category, category)
        results.append((display_name, prob, category))
        if cumulative >= threshold:
            break
    
    return results

# Интерфейс
st.title("📚 arXiv Статья Классификатор")
st.markdown("""
Определяет тематику научной статьи по **названию** и **аннотации**.
Модель обучена на 18,000+ статей из arXiv.org.
""")

# Ввод данных
st.subheader("Введите данные статьи")

title = st.text_input("📌 Название статьи *", placeholder="Например: Attention is All You Need")

abstract = st.text_area(
    "📄 Аннотация (необязательно)",
    placeholder="Введите аннотацию статьи здесь...",
    height=150
)

# Обработка пустого ввода
if title.strip() == "":
    st.warning("⚠️ Пожалуйста, введите название статьи")
    st.stop()

# Объединение текста
if abstract.strip():
    full_text = title + " [SEP] " + abstract
else:
    full_text = title

# Кнопка классификации
if st.button("🔍 Определить тематику", type="primary"):
    with st.spinner("Анализирую статью..."):
        try:
            model, tokenizer = load_model()
            predictions = predict(full_text, model, tokenizer)
            
            # Отображение результатов
            st.subheader("📊 Результаты классификации")
            
            # Прогресс-бары для вероятностей
            for display_name, prob, cat in predictions:
                st.markdown(f"**{display_name}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")
            
            # Пояснение
            if len(predictions) == 1:
                st.success(f"✅ Статья однозначно относится к категории **{predictions[0][0]}**")
            else:
                st.info(f"📌 Статья может относиться к нескольким областям (топ-{len(predictions)} категорий, суммарная вероятность > 95%)")
                
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            st.markdown("Попробуйте ввести другой текст или проверьте подключение.")

# Footer
st.markdown("---")
st.caption("Built with DistilBERT | Trained on arXiv papers | Deployed on Hugging Face Spaces")
