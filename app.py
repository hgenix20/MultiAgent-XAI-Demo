import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
#                          ENGINEER POLICY
##############################################################################

ENGINEER_POLICY = """
You are the Engineer. Focus on technical implementation and engineering tasks.
Keep your responses concise. If the request is unethical or out of scope, politely refuse.
"""

##############################################################################
#                          ANALYST POLICY
##############################################################################

ANALYST_POLICY = """
You are the Analyst. Focus on data-centric or machine learning approaches.
Keep your responses concise. If the request is unethical or out of scope, politely refuse.
"""

##############################################################################
#                          LOAD MODELS
##############################################################################

@st.cache_resource
def load_model_engineer():
    # Engineer: zephyr-7b-beta
    tokenizerE = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    modelE = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    return tokenizerE, modelE

@st.cache_resource
def load_model_analyst():
    # Analyst: phi-2
    tokenizerA = AutoTokenizer.from_pretrained("microsoft/Phi-2")
    modelA = AutoModelForCausalLM.from_pretrained("microsoft/Phi-2")
    return tokenizerA, modelA

# Load models
tokenizerE, modelE = load_model_engineer()
tokenizerA, modelA = load_model_analyst()

##############################################################################
#                     ENGINEER / ANALYST GENERATION
##############################################################################

def generate_engineer_response(engineer_policy, user_text, tokenizer, model):
    """
    Engineer sees:
      1) Its short policy
      2) Safe user text
    """
    prompt = f"""
{engineer_policy}

User text: {user_text}

Engineer, please provide a concise approach or solution. 
If out of scope/unethical, politely refuse.
"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=256,  # Extend length for detailed outputs
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_analyst_response(analyst_policy, user_text, engineer_output, tokenizer, model):
    """
    Analyst sees:
      1) Its short policy
      2) Safe user text
      3) Engineer's output, if relevant
    """
    prompt = f"""
{analyst_policy}

User text: {user_text}

Engineer's output: {engineer_output}

Analyst, please provide a concise approach or solution.
If out of scope/unethical, politely refuse.
"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=256,  # Extend length for detailed outputs
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

##############################################################################
#                         STREAMLIT APP
##############################################################################

st.title("Multi-Agent System with XAI Demo")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Enter a question/scenario:")

if st.button("Start/Continue Conversation"):
    if user_input.strip():
        # 1) Engineer
        engineer_resp = generate_engineer_response(
            engineer_policy=ENGINEER_POLICY,
            user_text=user_input,
            tokenizer=tokenizerE,
            model=modelE
        )
        st.session_state.conversation.append(("Engineer", engineer_resp))

        # 2) Analyst
        analyst_resp = generate_analyst_response(
            analyst_policy=ANALYST_POLICY,
            user_text=user_input,
            engineer_output=engineer_resp,
            tokenizer=tokenizerA,
            model=modelA
        )
        st.session_state.conversation.append(("Analyst", analyst_resp))

for speaker, text in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {text}")
