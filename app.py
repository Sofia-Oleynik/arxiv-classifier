import streamlit as st
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn import Softmax

# Настройка страницы
st.set_page_config(
    page_title="arXiv Article Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Кастомный CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-size: 1.2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("📚 arXiv Article Classifier")
st.markdown("*Определите тематику научной статьи по названию и аннотации*")
st.markdown("---")

# Категории
CATEGORY_NAMES = {
    'cs.AI': '🤖 Artificial Intelligence',
    'cs.CL': '💬 Computation & Language',
    'cs.CV': '👁️ Computer Vision',
    'cs.LG': '📊 Machine Learning',
    'physics': '⚛️ Physics',
    'math': '📐 Mathematics',
    'q-bio': '🧬 Quantitative Biology'
}

@st.cache_resource
def load_model():
    """Загрузка модели (кэшируется после первого вызова)"""
    try:
        # Замените на ваш репозиторий на Hugging Face
        model_name = "sofia-ol/arxiv-classifier"
        
        with st.spinner("📦 Загрузка модели... (первый запуск может занять 10-15 секунд)"):
            model = DistilBertForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

def predict(text, model, tokenizer, threshold=0.95):
    """Предсказание с топ-95% вероятностей"""
    if model is None or tokenizer is None:
        return []
    
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy().squeeze()
    
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

# Основной интерфейс
col1, col2 = st.columns([2, 1])

with col1:
    title = st.text_area(
        "📌 **Название статьи** *",
        placeholder="Пример: Attention is All You Need",
        height=100
    )
    
    abstract = st.text_area(
        "📄 **Аннотация** (необязательно)",
        placeholder="Введите аннотацию статьи здесь...",
        height=200
    )

with col2:
    st.markdown("### 🎯 Настройки")
    threshold = st.slider(
        "Порог уверенности",
        min_value=0.5,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Показывать категории, пока суммарная вероятность не превысит этот порог"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Поддерживаемые категории")
    for cat, name in CATEGORY_NAMES.items():
        st.markdown(f"- {name} (`{cat}`)")

# Кнопка классификации
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_btn = st.button("🔍 КЛАССИФИЦИРОВАТЬ", type="primary", use_container_width=True)

if classify_btn:
    if not title.strip():
        st.error("❌ Пожалуйста, введите название статьи")
    else:
        full_text = title
        if abstract.strip():
            full_text = title + " [SEP] " + abstract
        
        with st.spinner("🧠 Анализирую статью..."):
            model, tokenizer = load_model()
            predictions = predict(full_text, model, tokenizer, threshold)
            
            if predictions:
                st.markdown("---")
                st.subheader("📊 Результаты классификации")
                
                # Отображение результатов
                for i, (display_name, prob, category) in enumerate(predictions):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{display_name}**")
                            st.progress(float(prob), text=f"{prob*100:.1f}% уверенность")
                        with col2:
                            st.caption(f"`{category}`")
                
                # Пояснение
                st.markdown("---")
                if len(predictions) == 1:
                    st.success(f"✅ Статья однозначно относится к категории **{predictions[0][0]}**")
                else:
                    st.info(f"📌 Статья охватывает {len(predictions)} области (суммарная вероятность > {threshold*100:.0f}%)")
            else:
                st.warning("⚠️ Не удалось получить предсказания. Проверьте подключение к модели.")

# Footer
st.markdown("---")
st.caption("Built with DistilBERT | Fine-tuned on arXiv papers | Deployed with Streamlit")
