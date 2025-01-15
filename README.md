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

## Overview
This project demonstrates a Multi-Agent system using the `microsoft/Phi-3-mini-4k-instruct` model to simulate collaboration between an Engineer and an Analyst. The system is designed to generate technical solutions and complementary data-driven recommendations for user queries.

## Features
- Uses the `microsoft/Phi-3-mini-4k-instruct` model for natural language understanding.
- Simulates dialogue between two agents (Engineer and Analyst).
- Provides a summarized actionable plan at the end of the interaction.
- Built with Streamlit for an interactive user interface.

## Requirements
- Python 3.8 or higher
- Streamlit
- Transformers library
- PyTorch
- CUDA-enabled GPU for optimal performance (optional but recommended)

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
- **Hugging Face Transformers**: Using `AutoTokenizer`and  `AutoModelForCausalLM` with the `microsoft/phi-3-mini-4k-instruct` model for text generation.
- **Torch**

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Git LFS for handling large files:
   ```bash
   git lfs install
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Model Details
The `microsoft/Phi-3-mini-4k-instruct` model is a lightweight instruction-tuned language model optimized for efficient and concise responses. It is used here to:
- Generate technical solutions from the Engineer.
- Provide data-driven insights from the Analyst.

## Troubleshooting
- **Model Loading Issues:** Ensure all dependencies are installed and that your environment supports the `microsoft/Phi-3-mini-4k-instruct` model.
- **Performance Issues:** Use a CUDA-enabled GPU for better performance. If unavailable, ensure sufficient CPU resources.

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.