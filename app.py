import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
#                          LOAD MODELS
##############################################################################

@st.cache_resource
def load_model_engineer():
    # Engineer: DistilGPT-2
    tokenizerE = AutoTokenizer.from_pretrained("distilgpt2")
    modelE = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizerE, modelE

@st.cache_resource
def load_model_analyst():
    # Analyst: GPT-Neo-125M
    tokenizerA = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    modelA = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    return tokenizerA, modelA

# Load models
tokenizerE, modelE = load_model_engineer()
tokenizerA, modelA = load_model_analyst()

##############################################################################
#                     ENGINEER / ANALYST GENERATION
##############################################################################

def generate_engineer_response(user_text, tokenizer, model):
    """
    Engineer generates a concise approach or solution based on user input.
    """
    prompt = f"""
User text: {user_text}

Provide a technical approach or solution.
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

def generate_analyst_response(user_text, engineer_output, tokenizer, model):
    """
    Analyst provides an approach or solution based on user input and engineer's output.
    """
    prompt = f"""
User text: {user_text}

Engineer provided the following: {engineer_output}

Provide an approach or solution from a data-centric perspective.
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
            user_text=user_input,
            tokenizer=tokenizerE,
            model=modelE
        )
        st.session_state.conversation.append(("Engineer", engineer_resp))

        # 2) Analyst
        analyst_resp = generate_analyst_response(
            user_text=user_input,
            engineer_output=engineer_resp,
            tokenizer=tokenizerA,
            model=modelA
        )
        st.session_state.conversation.append(("Analyst", analyst_resp))

for speaker, text in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {text}")
