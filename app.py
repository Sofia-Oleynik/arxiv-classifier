import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn import Softmax
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="arXiv Classifier",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "./best_model"
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    
    indices = np.argsort(probs)[::-1]
    cumulative = 0
    results = []
    
    for idx in indices:
        prob = probs[idx]
        cumulative += prob
        category = model.config.id2label[idx]
        
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

st.title("📚 arXiv Статья Классификатор")
st.markdown("""
Определяет тематику научной статьи по **названию** и **аннотации**.
Модель обучена на 18,000+ статей из arXiv.org.
""")

st.subheader("Введите данные статьи")

title = st.text_input("📌 Название статьи *", placeholder="Например: Attention is All You Need")

abstract = st.text_area(
    "📄 Аннотация (необязательно)",
    placeholder="Введите аннотацию статьи здесь...",
    height=150
)

if title.strip() == "":
    st.warning("⚠️ Пожалуйста, введите название статьи")
    st.stop()

if abstract.strip():
    full_text = title + " [SEP] " + abstract
else:
    full_text = title

if st.button("🔍 Определить тематику", type="primary"):
    with st.spinner("Анализирую статью..."):
        try:
            model, tokenizer = load_model()
            predictions = predict(full_text, model, tokenizer)
            
            st.subheader("📊 Результаты классификации")
            
            for display_name, prob, cat in predictions:
                st.markdown(f"**{display_name}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")
            
            if len(predictions) == 1:
                st.success(f"✅ Статья однозначно относится к категории **{predictions[0][0]}**")
            else:
                st.info(f"📌 Статья может относиться к нескольким областям (топ-{len(predictions)} категорий, суммарная вероятность > 95%)")
                
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            st.markdown("Попробуйте ввести другой текст или проверьте подключение.")

st.markdown("---")
st.caption("Built with DistilBERT | Trained on arXiv papers | Deployed on Hugging Face Spaces")
