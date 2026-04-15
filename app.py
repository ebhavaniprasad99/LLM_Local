import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍕 Pizza Restaurant Assistant",
    page_icon="🍕",
    layout="centered"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0f0f0f;
        color: #f0ede6;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #f0ede6;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero h1 span {
        color: #ff6b35;
    }
    .hero p {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Chat messages */
    .msg-user {
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 12px 12px 4px 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        margin-left: 15%;
        color: #f0ede6;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .msg-bot {
        background: #1a1a1a;
        border: 1px solid #ff6b35;
        border-left: 3px solid #ff6b35;
        border-radius: 4px 12px 12px 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        margin-right: 15%;
        color: #f0ede6;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .msg-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .label-user { color: #888; }
    .label-bot  { color: #ff6b35; }

    /* Sources */
    .sources-box {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        margin-right: 15%;
        font-size: 0.8rem;
        color: #666;
    }
    .sources-box strong {
        color: #444;
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .source-item {
        margin-top: 0.3rem;
        padding: 0.3rem 0;
        border-top: 1px solid #1e1e1e;
        color: #555;
        font-size: 0.78rem;
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        padding: 0.8rem;
        background: #141414;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #1e1e1e;
    }
    .stat {
        text-align: center;
    }
    .stat-num {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        color: #ff6b35;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Input area */
    .stTextInput > div > div > input {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #f0ede6 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.8rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff6b35 !important;
        box-shadow: 0 0 0 1px #ff6b35 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #ff6b35 !important;
        color: #0f0f0f !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #ff8555 !important;
        transform: translateY(-1px) !important;
    }

    /* Clear button */
    .clear-btn > button {
        background: transparent !important;
        color: #444 !important;
        border: 1px solid #2a2a2a !important;
        font-size: 0.8rem !important;
    }
    .clear-btn > button:hover {
        color: #888 !important;
        border-color: #444 !important;
        transform: none !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #ff6b35 !important;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Scrollable chat area */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #333;
    }
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .empty-state p {
        font-size: 0.9rem;
        color: #444;
    }

    /* Suggestion chips */
    .chip {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        color: #666;
        margin: 0.2rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ─── LLM Setup (cached so it loads only once) ────────────────────────────────
@st.cache_resource
def load_chain():
    model = OllamaLLM(model="llama3.2")
    template = """
    You are an expert in answering questions about a pizza restaurant.
    Give helpful, concise answers based on the reviews provided.
    If the reviews don't contain relevant info, say so honestly.

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

# ─── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🍕 Pizza <span>Assistant</span></h1>
    <p>Ask anything about our restaurant — powered by real customer reviews</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats Bar ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <div class="stat">
        <div class="stat-num">{st.session_state.total_queries}</div>
        <div class="stat-label">Questions Asked</div>
    </div>
    <div class="stat">
        <div class="stat-num">{len(st.session_state.messages) // 2}</div>
        <div class="stat-label">Exchanges</div>
    </div>
    <div class="stat">
        <div class="stat-num">5</div>
        <div class="stat-label">Reviews per Query</div>
    </div>
    <div class="stat">
        <div class="stat-num">llama3.2</div>
        <div class="stat-label">Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Chat History ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🍕</div>
        <p>No questions yet. Try asking something like:</p>
        <div style="margin-top:1rem;">
            <span class="chip">What's the best pizza?</span>
            <span class="chip">How's the service?</span>
            <span class="chip">Good for families?</span>
            <span class="chip">Recommend a topping?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-label label-user">You</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">
                <div class="msg-label label-bot">🍕 Assistant</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

            # Show source reviews
            if "sources" in msg and msg["sources"]:
                with st.expander("📄 View source reviews used", expanded=False):
                    for i, doc in enumerate(msg["sources"]):
                        rating = doc.metadata.get("rating", "N/A")
                        date = doc.metadata.get("date", "N/A")
                        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        st.markdown(f"""
                        <div class="source-item">
                            ⭐ <strong>Rating:</strong> {rating} &nbsp;|&nbsp; 
                            📅 <strong>Date:</strong> {date}<br/>
                            <span style="color:#555;">{preview}</span>
                        </div>
                        """, unsafe_allow_html=True)

# ─── Input Area ───────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([6, 1.2, 1])

with col1:
    question = st.text_input(
        label="question",
        placeholder="Ask about our pizza, service, atmosphere...",
        label_visibility="collapsed",
        key="input_box"
    )

with col2:
    send = st.button("Ask 🍕", use_container_width=True)

with col3:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    clear = st.button("Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Handle Clear ─────────────────────────────────────────────────────────────
if clear:
    st.session_state.messages = []
    st.session_state.total_queries = 0
    st.rerun()

# ─── Handle Send ──────────────────────────────────────────────────────────────
if send and question.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    st.session_state.total_queries += 1

    # Get response
    with st.spinner("Searching reviews and generating answer..."):
        try:
            chain = load_chain()

            # Retrieve relevant reviews
            source_docs = retriever.invoke(question)

            # Generate answer
            result = chain.invoke({
                "reviews": source_docs,
                "question": question
            })

            # Add bot message with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "sources": source_docs
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {str(e)}\n\nMake sure Ollama is running with: `ollama serve`",
                "sources": []
            })

    st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1rem; 
            border-top:1px solid #1e1e1e; color:#333; font-size:0.75rem;">
    Powered by LangChain · ChromaDB · Ollama llama3.2 · Streamlit
</div>
""", unsafe_allow_html=True)
