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
    As an Engineer, generate a concise approach or solution based on user input.
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
    As an Analyst, provide an approach or solution based on user input and engineer's output.
    """
    prompt = f"""
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

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.text_area("User Input:", value=st.session_state.user_input, height=100, max_chars=None, key="user_input")

if st.button("Start/Continue Conversation"):
    if st.session_state.user_input.strip():
        user_text = st.session_state.user_input
        st.session_state.conversation.append(("User", user_text))

        # Engineer generates a response
        engineer_resp = generate_engineer_response(
            user_text=user_text,
            tokenizer=tokenizerE,
            model=modelE
        )
        st.session_state.conversation.append(("Engineer", engineer_resp))

        # Analyst generates a response based on engineer's output
        analyst_resp = generate_analyst_response(
            user_text=user_text,
            engineer_output=engineer_resp,
            tokenizer=tokenizerA,
            model=modelA
        )
        st.session_state.conversation.append(("Analyst", analyst_resp))

        # Limit the conversation to 3 exchanges between Engineer and Analyst
        for _ in range(2):
            engineer_resp = generate_engineer_response(
                user_text=analyst_resp,
                tokenizer=tokenizerE,
                model=modelE
            )
            st.session_state.conversation.append(("Engineer", engineer_resp))

            analyst_resp = generate_analyst_response(
                user_text=engineer_resp,
                engineer_output=engineer_resp,
                tokenizer=tokenizerA,
                model=modelA
            )
            st.session_state.conversation.append(("Analyst", analyst_resp))

for speaker, text in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**{speaker}:** {text}")
    else:
        st.markdown(f"<div style='display:none'>{speaker}: {text}</div>", unsafe_allow_html=True)
