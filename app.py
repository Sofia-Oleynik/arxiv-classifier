import streamlit as st
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn import Softmax

st.set_page_config(
    page_title="arXiv Classifier",
    page_icon="📚",
    layout="centered"
)

st.title("📚 arXiv Article Classifier")
st.markdown("Определите тематику научной статьи по названию и аннотации")

CATEGORY_NAMES = {
    'cs.AI': '🤖 Artificial Intelligence',
    'cs.CL': '💬 Computation & Language',
    'cs.CV': '👁️ Computer Vision',
    'physics': '⚛️ Physics',
    'math': '📐 Mathematics',
    'q-bio': '🧬 Quantitative Biology'
}

@st.cache_resource
def load_model():
    """
    Модель загружается один раз при первом вызове
    и сохраняется в кэше Streamlit.
    """
    try:
        model_name = "sofia-ol/arxiv-classifier" 
        
        with st.spinner("📦 Загрузка модели... "):
            model = DistilBertForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
            
            # Перемещаем на GPU если доступно
            if torch.cuda.is_available():
                model = model.cuda()
            
        st.success("✅ Модель успешно загружена!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None, None

def predict_article(text, model, tokenizer, threshold=0.95):
    """Предсказание категории с топ-95% вероятностей"""
    if model is None or tokenizer is None:
        return [("⚠️ Модель не загружена", 1.0)]
    
    # Определяем устройство модели
    device = next(model.parameters()).device
    
    # Токенизация
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding=True
    )
    
    # Перемещаем на то же устройство, что и модель
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Инференс
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy().squeeze()
    
    # Сортировка и накопление до 95%
    indices = np.argsort(probs)[::-1]
    cumulative = 0
    results = []
    
    for idx in indices:
        prob = probs[idx]
        cumulative += prob
        category = model.config.id2label[idx]
        display_name = CATEGORY_NAMES.get(category, category)
        results.append((display_name, prob, category))
        
        if cumulative >= threshold:
            break
    
    return results

# ============================================
# ИНТЕРФЕЙС
# ============================================

# Поля ввода
title = st.text_area(
    "📌 **Название статьи** *",
    placeholder="Пример: Attention is All You Need",
    height=100
)

abstract = st.text_area(
    "📄 **Аннотация** (необязательно)",
    placeholder="Введите аннотацию здесь...",
    height=200
)

# Слайдер для порога уверенности
threshold = st.slider(
    "🎯 Порог уверенности",
    min_value=0.5,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="Показывать категории, пока суммарная вероятность не превысит этот порог"
)

# Кнопка классификации
if st.button("🔍 Классифицировать", type="primary"):
    if not title.strip():
        st.error("❌ Пожалуйста, введите название статьи")
    else:
        # Объединяем текст
        full_text = title
        if abstract.strip():
            full_text = title + " [SEP] " + abstract
        
        # Показываем спиннер только во время предсказания (модель уже загружена!)
        with st.spinner("🧠 Анализирую статью..."):
            # Модель загружается здесь, но ТОЛЬКО ПРИ ПЕРВОМ ЗАПРОСЕ
            model, tokenizer = load_model()
            predictions = predict_article(full_text, model, tokenizer, threshold)
            
            if predictions:
                st.subheader("📊 Результаты классификации")
                
                # Отображаем результаты с прогресс-барами
                for display_name, prob, category in predictions:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{display_name}**")
                        st.progress(float(prob), text=f"{prob*100:.1f}%")
                    with col2:
                        st.caption(f"`{category}`")
                
                # Пояснение
                if len(predictions) == 1:
                    st.success(f"✅ Статья однозначно относится к категории **{predictions[0][0]}**")
                else:
                    st.info(f"📌 Статья относится к {len(predictions)} областям (суммарная вероятность > {threshold*100:.0f}%)")
            else:
                st.warning("⚠️ Не удалось получить предсказания")

# Footer
st.markdown("---")
st.caption("Built with DistilBERT | Fine-tuned on arXiv papers | Deployed with Streamlit")

# Информация о статусе модели (для отладки)
with st.expander("ℹ️ Информация о модели"):
    st.write("Модель загружается один раз при первом запросе и кэшируется")
    st.write("Благодаря `@st.cache_resource` последующие запросы работают мгновенно")
