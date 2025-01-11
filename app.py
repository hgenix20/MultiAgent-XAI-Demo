import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_prompt(conversation, agent_name):
    """
    Construct a single prompt that includes the entire conversation so far,
    labeling each line with speaker, and ends with the new agent's label.
    """
    text_blocks = []
    for speaker, text in conversation:
        text_blocks.append(f"{speaker}: {text}")

    # Now add the new agent's label at the end, so the model continues from there
    text_blocks.append(f"{agent_name}:")
    return "\n".join(text_blocks)

def generate_response(agent_name, model, tokenizer, conversation):
    """
    Takes the entire conversation as context, plus the agent name, 
    and runs a single inference call for that agent.
    """
    prompt_text = build_prompt(conversation, agent_name)
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@st.cache_resource
def load_agentA():
    """Loads the DistilGPT2 model/tokenizer for Agent A."""
    tokenizerA = AutoTokenizer.from_pretrained("distilgpt2")
    modelA = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizerA, modelA

@st.cache_resource
def load_agentB():
    """Loads the GPT-Neo-125M model/tokenizer for Agent B."""
    tokenizerB = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    modelB = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    return tokenizerB, modelB

# Load agents
tokenizerA, modelA = load_agentA()
tokenizerB, modelB = load_agentB()

# Streamlit app starts here
st.title("True Multi-Agent Conversation")

# We store the conversation as a list of (speaker, text).
if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Enter a question or scenario:")

if st.button("Start/Continue Conversation"):
    # 1) If this is the first message, add the user input
    if len(st.session_state.conversation) == 0:
        st.session_state.conversation.append(("User", user_input))
    else:
        # If conversation is ongoing, append user’s new input
        st.session_state.conversation.append(("User", user_input))

    # --- AGENT A Step ---
    agentA_text = generate_response(
        agent_name="Agent A",
        model=modelA,
        tokenizer=tokenizerA,
        conversation=st.session_state.conversation
    )
    st.session_state.conversation.append(("Agent A", agentA_text))

    # --- AGENT B Step ---
    agentB_text = generate_response(
        agent_name="Agent B",
        model=modelB,
        tokenizer=tokenizerB,
        conversation=st.session_state.conversation
    )
    st.session_state.conversation.append(("Agent B", agentB_text))

# Display the entire conversation so far
for speaker, text in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {text}")