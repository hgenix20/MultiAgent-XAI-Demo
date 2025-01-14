import streamlit as st

from transformers import pipeline

##############################################################################
#                          LOAD MODELS
##############################################################################

@st.cache_resource
def load_model_engineer():
    # Engineer: Microsoft PHI-4 via pipeline
    engineer_pipeline = pipeline(
        "text-generation",
        model="microsoft/phi-4",
        trust_remote_code=True
    )
    return engineer_pipeline

@st.cache_resource
def load_model_analyst():
    # Analyst: Microsoft PHI-4 via pipeline
    analyst_pipeline = pipeline(
        "text-generation",
        model="microsoft/phi-4",
        trust_remote_code=True
    )
    return analyst_pipeline

# Load models
engineer_pipeline = load_model_engineer()
analyst_pipeline = load_model_analyst()

##############################################################################
#                     ENGINEER / ANALYST GENERATION
##############################################################################

def generate_response(prompt, pipeline_model, max_sentences=2):
    """
    Generate a concise response based on the provided prompt.
    """
    outputs = pipeline_model(prompt, max_new_tokens=50, temperature=0.6, top_p=0.8)
    response = outputs[0]["generated_text"].strip()
    # Limit to max_sentences by splitting and rejoining
    return " ".join(response.split(".")[:max_sentences]) + "."

def summarize_conversation(conversation):
    """
    Summarize the entire conversation to produce a cohesive and actionable plan.
    """
    summary = "### Final Plan\n"
    key_points = []
    for speaker, text in conversation:
        if speaker == "Engineer" or speaker == "Analyst":
            key_points.append(f"- {speaker}: {text}")
    summary += "\n".join(key_points[-6:])  # Include only the last 3 turns each
    summary += "\n\nThis collaborative plan integrates technical and analytical insights into an actionable framework."
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

        engineer_prompt_base = f"Given the problem: {user_text}, provide a concise and actionable technical solution."
        analyst_prompt_base = "Based on the engineer's suggestion, provide complementary data-driven recommendations."

        for turn in range(3):
            # Engineer generates a response
            with st.spinner(f"Engineer is formulating response {turn + 1}..."):
                engineer_resp = generate_response(
                    prompt=engineer_prompt_base,
                    pipeline_model=engineer_pipeline
                )
                st.session_state.conversation.append(("Engineer", engineer_resp))

            # Display Engineer response
            st.markdown(f"### Engineer Response ({turn + 1})\n{engineer_resp}")

            # Analyst generates a response based on engineer's output
            with st.spinner(f"Analyst is formulating response {turn + 1}..."):
                analyst_resp = generate_response(
                    prompt=f"Engineer suggested: {engineer_resp}. {analyst_prompt_base}",
                    pipeline_model=analyst_pipeline
                )
                st.session_state.conversation.append(("Analyst", analyst_resp))

            # Display Analyst response
            st.markdown(f"### Analyst Response ({turn + 1})\n{analyst_resp}")

        # Summarize the final plan
        with st.spinner("Generating the final plan..."):
            final_plan = summarize_conversation(st.session_state.conversation)
            st.session_state.conversation.append(("Summary", final_plan))
            st.markdown(final_plan)