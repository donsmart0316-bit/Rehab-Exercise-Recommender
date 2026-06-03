import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import shutil
import profile

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.neighbors import NearestNeighbors
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from fpdf import FPDF

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main {background-color: #cbd3da;}
    .stButton>button {background-color: #2E8B57; color: white; border-radius: 10px;}
    .card {background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 20px;}
    h1, h2, h3 {color: #1f4d3d;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Personalized Rehab Exercise Recommender")
st.markdown("**Safe • Personalized • Expert-Guided**")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_core_data(show_spinner=False):
    df = pd.read_excel("rehab_recommender.xlsx")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df.fillna("")

core_df = load_core_data()

# ---------------- MODELS ----------------
@st.cache_resource (show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedder = load_embedding_model()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Use a more reliable model + Chat interface
@st.cache_resource (show_spinner=False)
def load_llm():
    # Option 1: Most reliable right now (Recommended)
    return ChatGroq(
        model_name="llama-3.1-8b-instant",          # Very stable for text-generation
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=950,
    )
llm = load_llm()

# ====================== RAG (Textbooks) ======================
@st.cache_resource (show_spinner=False)
def load_rag_vectorstore():
    vectorstore_path = "textbook_vectorstore"
    if os.path.exists(vectorstore_path):
       return FAISS.load_local(vectorstore_path, embedder, allow_dangerous_deserialization=True)  # Clear old vectorstore to ensure fresh processing
    
    st.info("Building textbook vector database (first run only)...")
    documents = []
    textbook_folder = "textbooks"   # Change if needed
    
    for filename in os.listdir(textbook_folder):
        if filename.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(os.path.join(textbook_folder, filename))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
                continue  # Skip files that fail to load

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250, separators=["\n\n", "\n",".", " ", ""])
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(vectorstore_path)
    st.success(f"✅ Processed {len(chunks)} textbook chunks")
    return vectorstore
    
vectorstore = load_rag_vectorstore() 
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 20})

# ====================== SMART QUERY BUILDER ======================

def build_rag_query(condition, goal, symptoms):

    condition_lower = condition.lower()

    # =====================================================
    # CLINICAL ENRICHMENT
    # =====================================================

    if "headache" in condition_lower:

        condition += """
        cervicogenic headache
        tension headache
        cervical spine dysfunction
        neck posture
        upper trapezius tightness
        deep neck flexors
        headache physiotherapy
        posture correction
        """

    elif "knee" in condition_lower:

        condition += """
        knee rehabilitation
        quadriceps strengthening
        patellofemoral pain
        ACL rehab
        meniscus rehab
        proprioception
        gait retraining
        """

    elif "stroke" in condition_lower:

        condition += """
        neurological rehabilitation
        gait training
        balance training
        coordination exercises
        neuroplasticity
        stroke physiotherapy
        mobility recovery
        """

    elif "shoulder" in condition_lower:

        condition += """
        rotator cuff rehabilitation
        scapular stabilization
        shoulder mobility
        impingement syndrome
        shoulder physiotherapy
        """

    elif "low back" in condition_lower or "back pain" in condition_lower:

        condition += """
        lumbar stabilization
        core strengthening
        lumbar mobility
        posture correction
        low back rehabilitation
        spine physiotherapy
        """
    elif "ankle" in condition_lower:

        condition += """
        ankle sprain rehabilitation
        balance training
        proprioception
        ankle strengthening
        gait mechanics
        """

    elif "neck" in condition_lower:

        condition += """
        cervical rehabilitation
        posture correction
        neck mobility
        deep neck flexor training
        cervical stabilization
        """

    # =====================================================
    # FINAL QUERY
    # =====================================================

    query = f"""
    {condition}

    symptoms:
    {symptoms}

    rehabilitation goals:
    {goal}

    rehabilitation exercises
    physiotherapy management
    therapeutic exercise
    progression plan
    precautions
    contraindications
    pain management
    strengthening
    stretching
    mobility training
    functional recovery
    home exercise program
    safety guidelines
    """

    return query


# ====================== SIDEBAR & INPUTS ======================
st.subheader("Your Information")

col1, col2 = st.columns([2, 1])
with col1:
    condition = st.text_input("**Main Condition**", placeholder="Knee Osteoarthritis")
    goal = st.selectbox("**Your Main Goal**", ["Pain relief", "Mobility", "Strength", "Balance", "Return to sport"])
    specific_symptoms = st.text_area("**Specific Symptoms**", placeholder="Swelling, stiffness, instability, numbness...")

with col2:
    age = st.number_input("**Age**", 18, 90, 45)
    gender = st.selectbox("**Gender**", ["Male", "Female"])
    pain_level = st.slider("**Pain Level (1-10)**", 1, 10, 5)

st.subheader("Important History & Details")
col_a, col_b = st.columns(2)
with col_a:
    time_since_injury = st.selectbox("Time Since Injury/Onset", ["Acute (<2 weeks)", "Subacute (2-6 weeks)", "Chronic (>6 weeks)"])
    bmi = st.number_input("**BMI**", 15.0, 45.0, 25.0)
    condition_duration = st.selectbox("**Condition Duration**", ["First episode", "Recurrent", "Long-term (>1 year)"])
    daily_activity = st.selectbox("**Daily Activity Level**", ["Sedentary", "Light", "Moderate", "Active"])

with col_b:
    recent_surgery = st.selectbox("Recent Surgery?", ["No", "Yes - Within 3 months", "Yes - Older"])
    high_bp = st.selectbox("High Blood Pressure?", ["No", "Yes - Controlled", "Yes - Uncontrolled"])
    dizziness = st.selectbox("Dizziness / Balance Issues?", ["No", "Occasional", "Frequent"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes - Controlled", "Yes - Uncontrolled"])

equipment = st.multiselect("**Equipment Available at Home**", 
    ["None", "Chair", "Resistance Band", "Towel", "Dumbbells", "Mat", "Wall", "Hot Water (Heating)", "Ice Pack / Cold Compress (Cooling)", "Heating Pad", "Foam Roller"])

previous_response = st.text_area("Previous Exercises Tried & Response", placeholder="e.g. Straight leg raise helped...")

def valid_conditions(text):
    if not text:
        st.warning("Please enter a valid condition.")
        return False
    if len(text) < 3:
        st.warning("Condition must be at least 3 characters long.")
        return False
    if sum(c.isalpha() or c.isspace() for c in text) / len(text) < 0.7:  # At least 70% should be letters/spaces
        st.warning("Condition must contain mostly letters and spaces.")
        return False
    junk_text = ["asdf", "qwer", "zxcv", "lorem", "ipsum", "test", "1234"]
    if text.lower() in junk_text:
        st.warning("Please enter a valid condition.")
        return False
    return True

# ====================== CLINICAL VETTING ENGINE ======================
def extract_textbook_exercises(text):
    prompt = f"""
Extract ONLY physiotherapy exercises, techniques, and interventions from this text.

TEXT:
{text}

Return as bullet points only."""
    return llm.invoke(prompt).content

def get_dataset_exercises(condition):
    filtered = core_df[core_df["condition"].str.lower().str.contains(condition.lower(), na=False)]
    if filtered.empty:
        return "No direct matches in dataset."
    return "\n".join([f"- {r.get('exercise_name','Unknown')} | {r.get('benefits','')}" for _, r in filtered.iterrows()])

def clinical_vet(profile, dataset_ex, textbook_ex):
    prompt = f"""
You are an expert (consultant) physiotherapist doing clinical vetting.
You are speaking DIRECTLY to the patient as "you" and "your" and try as much as possible to tailor treatment to Equipment-based modifications
You have already comprehensively reviewed all clinical knowledge internally and the patient's profile information
DO NOT mention the process or the textbook, sources, analysis ordataset. Just speak directly to the patient and give them a safe, effective, personalized plan based on all the information you have.

TEXTBOOK:
{textbook_ex}

DATASET:
{dataset_ex}

PATIENT:
{profile}

Rules:
- Look very comprehensively at textbook knowledge
- Remove unsafe or irrelevant items
- Rank top 5 most suitable exercises
- Be safe and conservative

Output Format:
Safety Summary (highlight any risks)
Top 5 Recommended Exercises:
Instructions & Reps/sets for each exercise:
Precautions for you:
Weekly Progression:
"""
    return llm.invoke(prompt).content

# ====================== GENERATE PLAN ======================
if st.button("🚀 Generate Hybrid Personalized Plan", type="primary", use_container_width=True, key="generate_btn"):
    if not valid_conditions(condition):
        st.error("Please enter a valid condition to generate a plan.")
        st.stop()
    with st.spinner("Creating your plan..."):
        try:
            # Core filtering
            filtered = core_df[core_df['condition'].str.lower().str.contains(condition.lower(), na=False)]
            if pain_level >= 7:
                filtered = filtered[filtered.get('difficulty', 'Beginner').str.contains('Beginner', case=False, na=True)]
            if filtered.empty:
                filtered = core_df.sample(min(15, len(core_df)))  # Show random exercises as fallback

            # RAG from textbooks
            rag_query = build_rag_query(condition, goal, specific_symptoms)
            textbook_docs = retriever.invoke(rag_query)
            textbook_text = "\n\n".join([doc.page_content[:1200] for doc in textbook_docs])
                    
            # Extract exercises 
            textbook_exercises = extract_textbook_exercises(textbook_text)
            dataset_exercises = get_dataset_exercises(condition)
    
            # Strong LLM Prompt with ALL variables
            prompt = f"""You are an expert physiotherapist. Speak directly to the user as "you" and "your". Be very safe and detailed.

**Your Profile:**
- Age: {age} | Gender: {gender} | BMI: {bmi}
- Condition: {condition} | Duration: {condition_duration} | Time since onset: {time_since_injury}
- Pain Level: {pain_level}/10 | Symptoms: {specific_symptoms}
- Goal: {goal} | Daily Activity: {daily_activity}
- Recent Surgery: {recent_surgery}
- High Blood Pressure: {high_bp}
- Dizziness/Balance Issues: {dizziness}
- Previous Exercises Response: {previous_response}
- Equipment Available: {equipment}

**Relevant Textbook Knowledge:**
{textbook_text}

Create a complete personalized plan for **you**:
1. Safety Summary (highlight any risks)
2. Top 3-5 Recommended Exercises with customized instructions, reps/sets, precautions for **you** and ***your* condition**
3. Weekly Progression Plan (Week 1-2, Week 3-4, Week 5+)
4. Equipment-based modifications
Keep responses practical and concise. Always prioritize safety and patient-specific factors."""

            final_plan = clinical_vet(profile, dataset_exercises, textbook_exercises)

            st.session_state.final_plan = final_plan
            st.session_state.condition = condition
            
            st.success("✅ Plan Generated Successfully")
            st.markdown(final_plan)

            # ====================== AUTO SAVE ======================
            plan_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "condition": condition,
                "age": age,
                "pain_level": pain_level,
                "plan": final_plan
            }
            save_path = "saved_plans.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    all_plans = json.load(f)
            else:
                all_plans = []
            all_plans.append(plan_data)
            with open(save_path, "w") as f:
                json.dump(all_plans, f, indent=2)

            # PDF Download (Fixed)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Rehab Plan\nCondition: {condition}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n{final_plan}")
            pdf.output("Rehab_plan.pdf")

            with open("Rehab_plan.pdf", "rb") as f:
                st.download_button(
                    label="📥 Download Plan as PDF", 
                    key = "download_pdf",
                    data=f,
                    file_name=f"Rehab_Plan_{condition.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error("Error generating plan. Please try again.")
            
# ====================== CHAT INTERFACE (Fixed) ======================
if "final_plan" in st.session_state:
    st.subheader("💬 Ask Questions About Your Plan")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask anything (modifications, progression, pain concerns...)")
    if user_question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_prompt = f"""Context from patient's rehab plan: {st.session_state.final_plan[:1800]}
                
Question: {user_question}
Answer as an expert physiotherapist, and using "you" to the user. Keep it concise, Be safe, helpful and specific."""
                answer = llm.invoke(chat_prompt).content
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
st.markdown("---")
st.markdown("**Dr Obasi Kizito (BPT, MRTB)**")  
st.markdown("📧 donsmart0316@gmail.com")
st.markdown("Need a consultation? Feel free to reach out to me directly.")
st.caption("Educational tool only • Always consult a qualified physiotherapist • Stop if pain increases")
