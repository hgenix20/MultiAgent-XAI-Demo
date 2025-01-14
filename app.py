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
    Provide a technical approach or solution that directly addresses the problem. Ensure your response is actionable and concise.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,  # Generate up to 80 new tokens
        temperature=0.7,
        do_sample=True,
        top_p=0.85,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    explanation = f"Engineer response based on user input: '{user_text}'"
    return tokenizer.decode(outputs[0], skip_special_tokens=True), explanation

def generate_analyst_response(user_text, engineer_output, tokenizer, model):
    """
    As an Analyst, provide an approach or solution based on user input and engineer's output.
    """
    prompt = f"""
Engineer provided the following: {engineer_output}

Based on this, provide an actionable data-driven approach or solution to complement the engineer's perspective.
"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,  # Generate up to 80 new tokens
        temperature=0.7,
        do_sample=True,
        top_p=0.85,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    explanation = f"Analyst response based on Engineer's output: '{engineer_output}'"
    return tokenizer.decode(outputs[0], skip_special_tokens=True), explanation

def summarize_conversation(conversation):
    """
    Summarize the entire conversation to produce a comprehensive plan.
    """
    summary = "**Final Plan:**\n"
    for speaker, text in conversation:
        if speaker != "User":
            summary += f"- **{speaker}:** {text}\n"
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

        # Engineer generates a response
        with st.spinner("Engineer is formulating a solution..."):
            engineer_resp, engineer_explanation = generate_engineer_response(
                user_text=user_text,
                tokenizer=tokenizerE,
                model=modelE
            )
            st.session_state.conversation.append(("Engineer", engineer_resp))
            st.session_state.conversation.append(("Engineer Explanation", engineer_explanation))

        # Display Engineer response immediately
        st.markdown(f"**Engineer:** {engineer_resp}")
        st.markdown(f"<i>{engineer_explanation}</i>", unsafe_allow_html=True)

        # Analyst generates a response based on engineer's output
        with st.spinner("Analyst is analyzing data and providing insights..."):
            analyst_resp, analyst_explanation = generate_analyst_response(
                user_text=user_text,
                engineer_output=engineer_resp,
                tokenizer=tokenizerA,
                model=modelA
            )
            st.session_state.conversation.append(("Analyst", analyst_resp))
            st.session_state.conversation.append(("Analyst Explanation", analyst_explanation))

        # Display Analyst response immediately
        st.markdown(f"**Analyst:** {analyst_resp}")
        st.markdown(f"<i>{analyst_explanation}</i>", unsafe_allow_html=True)

        # Limit the conversation to 3 exchanges between Engineer and Analyst
        for _ in range(2):
            with st.spinner("Engineer is formulating a solution..."):
                engineer_resp, engineer_explanation = generate_engineer_response(
                    user_text=analyst_resp,
                    tokenizer=tokenizerE,
                    model=modelE
                )
                st.session_state.conversation.append(("Engineer", engineer_resp))
                st.session_state.conversation.append(("Engineer Explanation", engineer_explanation))

                # Display Engineer response immediately
                st.markdown(f"**Engineer:** {engineer_resp}")
                st.markdown(f"<i>{engineer_explanation}</i>", unsafe_allow_html=True)

            with st.spinner("Analyst is analyzing data and providing insights..."):
                analyst_resp, analyst_explanation = generate_analyst_response(
                    user_text=engineer_resp,
                    engineer_output=engineer_resp,
                    tokenizer=tokenizerA,
                    model=modelA
                )
                st.session_state.conversation.append(("Analyst", analyst_resp))
                st.session_state.conversation.append(("Analyst Explanation", analyst_explanation))

                # Display Analyst response immediately
                st.markdown(f"**Analyst:** {analyst_resp}")
                st.markdown(f"<i>{analyst_explanation}</i>", unsafe_allow_html=True)

        # Generate the summary after the conversation
        final_plan = summarize_conversation(st.session_state.conversation)
        st.session_state.conversation.append(("Summary", final_plan))
        st.markdown(final_plan)

for speaker, text in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**{speaker}:** {text}")
    elif speaker == "Summary":
        st.markdown(f"**{speaker}:** {text}")
    elif "Explanation" in speaker:
        st.markdown(f"<i>{speaker}: {text}</i>", unsafe_allow_html=True)
    else:
        st.markdown(f"**{speaker}:** {text}")
