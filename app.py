import streamlit as st

from transformers import AutoModelForCausalLM
try:
    from transformers import AutoTokenizer
except ImportError:
    from transformers import GPT2Tokenizer as AutoTokenizer

##############################################################################
#                          LOAD MODELS
##############################################################################

@st.cache_resource
def load_model_engineer():
    # Engineer: EleutherAI GPT-Neo-125M
    tokenizerE = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizerE.pad_token is None:
        tokenizerE.add_special_tokens({'pad_token': '[PAD]'})
    modelE = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    return tokenizerE, modelE

@st.cache_resource
def load_model_analyst():
    # Analyst: Microsoft DialoGPT-small
    tokenizerA = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizerA.pad_token is None:
        tokenizerA.add_special_tokens({'pad_token': '[PAD]'})
    modelA = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizerA, modelA

# Load models
tokenizerE, modelE = load_model_engineer()
tokenizerA, modelA = load_model_analyst()

##############################################################################
#                     ENGINEER / ANALYST GENERATION
##############################################################################

def generate_engineer_response(user_text, tokenizer, model):
    """
    Generate a concise technical response from the Engineer based on the user's input.
    """
    prompt = f"""
User: {user_text}

Engineer: Provide a technical solution or approach that directly addresses the user's problem.
"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,  # Restrict length
        temperature=0.6,
        do_sample=True,
        top_p=0.8,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def generate_analyst_response(engineer_output, tokenizer, model):
    """
    Generate a conversational response from the Analyst based on the Engineer's response.
    """
    prompt = f"""
Engineer: {engineer_output}

Analyst: Respond with actionable insights or complementary data-driven recommendations.
"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,  # Restrict length
        temperature=0.6,
        do_sample=True,
        top_p=0.8,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def summarize_conversation(conversation):
    """
    Summarize the entire conversation to produce a cohesive and actionable plan.
    """
    summary = "### Final Plan\n"
    engineer_response = next((text for speaker, text in conversation if speaker == "Engineer"), "")
    analyst_response = next((text for speaker, text in conversation if speaker == "Analyst"), "")

    summary += f"- **Engineer Perspective:**\n  {engineer_response}\n\n"
    summary += f"- **Analyst Perspective:**\n  {analyst_response}\n\n"
    summary += "This collaborative plan ensures a balance of technical and analytical insights."

    return summary

##############################################################################
#                         STREAMLIT APP
##############################################################################

st.title("Multi-Agent System with XAI Demo")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.text_area("Enter your query:", value=st.session_state.user_input, height=100, max_chars=None, key="user_input")

if st.button("Generate Responses"):
    if st.session_state.user_input.strip():
        user_text = st.session_state.user_input
        st.session_state.conversation = [("User", user_text)]  # Clear and restart conversation

        # Engineer generates a response
        with st.spinner("Engineer is formulating a solution..."):
            engineer_resp = generate_engineer_response(
                user_text=user_text,
                tokenizer=tokenizerE,
                model=modelE
            )
            st.session_state.conversation.append(("Engineer", engineer_resp))

        # Display Engineer response immediately
        st.markdown(f"### Engineer Response\n{engineer_resp}")

        # Analyst generates a response based on engineer's output
        with st.spinner("Analyst is analyzing data and providing insights..."):
            analyst_resp = generate_analyst_response(
                engineer_output=engineer_resp,
                tokenizer=tokenizerA,
                model=modelA
            )
            st.session_state.conversation.append(("Analyst", analyst_resp))

        # Display Analyst response immediately
        st.markdown(f"### Analyst Response\n{analyst_resp}")

        # Summarize the final plan
        with st.spinner("Generating the final plan..."):
            final_plan = summarize_conversation(st.session_state.conversation)
            st.session_state.conversation.append(("Summary", final_plan))
            st.markdown(final_plan)