---
title: MultiAgent XAI Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: mit
short_description: A demo showcasing multi-agent conversational AI.
---

## MultiAgent XAI Demo

This demo leverages the `microsoft/phi-4` language model for simulating a conversation between two roles: an Engineer and an Analyst. The goal is to collaboratively address user-provided queries and produce actionable insights.

### Features
- **Engineer Role**: Provides concise, technical solutions.
- **Analyst Role**: Offers data-driven recommendations to complement the Engineer's response.
- **Natural Dialogue**: Facilitates a three-turn conversation between the roles.
- **Actionable Summary**: Generates a final plan summarizing key insights.

### How It Works
1. The user enters a query.
2. The Engineer and Analyst respond alternately, building on each other's inputs.
3. A final summary is generated, integrating technical and analytical perspectives.

### Technology Stack
- **Streamlit**: Interactive web interface.
- **Hugging Face Transformers**: Using `pipeline` with the `microsoft/phi-4` model for text generation.

### Getting Started
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

### Configuration Reference
Refer to the [Hugging Face Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference) for deployment options.

### License
This project is licensed under the MIT License. See the LICENSE file for details.
