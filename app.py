import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Multi-Agent Dialogue Simulator")
user_input = st.text_input("Enter a scenario or question:")

if st.button("Generate Collaboration"):
    # Create a custom prompt with two roles
    prompt = f"""
    The following is a conversation between two agents:
    Agent A: A Lean Six Sigma process re-engineer.
    Agent B: An AI/data scientist.

    They discuss how to solve the user's challenge:

    User scenario: {user_input}

    Agent A: Let's break down the problem step by step.
    Agent B:
    """

    # Generate the conversation
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=300,
        min_length=50, 
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
    )
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to split or isolate Agent B's portion
    # (For simplicity, we'll just display raw_text)
    st.markdown("**Conversation**:")
    st.write(raw_text)
