import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
#                          LOAD MODELS
##############################################################################

@st.cache_resource
def load_model_engineer():
    # Engineer: DistilGPT-2
    tokenizerE = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizerE.pad_token is None:
        tokenizerE.add_special_tokens({'pad_token': '[PAD]'})
    modelE = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizerE, modelE

@st.cache_resource
def load_model_analyst():
    # Analyst: GPT-Neo-125M
    tokenizerA = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizerA.pad_token is None:
        tokenizerA.add_special_tokens({'pad_token': '[PAD]'})
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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,  # Generate up to 100 new tokens
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id
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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,  # Generate up to 100 new tokens
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_conversation(conversation):
    """
    Summarize the entire conversation to produce a comprehensive plan.
    """
    summary = "Here is the summarized plan based on the discussion:\n"
    for speaker, text in conversation:
        if speaker != "User":
            summary += f"- {speaker}: {text}\n"
    return summary

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

        # Display "Thinking..."
        with st.spinner("Thinking..."):
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

            # Generate the summary after the conversation
            final_plan = summarize_conversation(st.session_state.conversation)
            st.session_state.conversation.append(("Summary", final_plan))

for speaker, text in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**{speaker}:** {text}")
    elif speaker == "Summary":
        st.markdown(f"**{speaker}:** {text}")
    else:
        st.markdown(f"<div style='display:none'>{speaker}: {text}</div>", unsafe_allow_html=True)
