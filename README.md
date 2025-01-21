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
short_description: A multi-agent conversational AI with explainability.
---

# MultiAgent XAI Demo

## Overview
The **Multi-Agent XAI Demo** is an advanced Streamlit-based web application designed to provide AI-powered technical solutions with built-in explainability. By integrating **Explainable AI (XAI)**, the system ensures transparency and interpretability, empowering users with actionable insights and a clear understanding of AI-generated recommendations.

This project demonstrates a Multi-Agent system using the `microsoft/Phi-3-mini-4k-instruct` model to simulate collaboration between an **Engineer** and an **Analyst**, generating technical solutions and complementary data-driven recommendations for user queries.

## Key Features

### User-Friendly Query Submission
- Intuitive interface for seamless query input
- Efficient processing for rapid AI-generated responses

### AI-Powered Insights
- **Engineer Agent:** Delivers precise technical solutions for complex challenges
- **Analyst Agent:** Provides complementary data-driven insights to enhance analysis

### Explainable AI (XAI)
- Every response includes a detailed explanation, offering clarity into the AI's reasoning and decision-making process

### Comprehensive Summarization
- The system compiles responses into an actionable plan, ensuring well-structured insights for decision-making

## Applications

### Industry Applications
- Predictive maintenance for manufacturing and industrial processes
- Process automation to optimize workflows
- Resource allocation and operational efficiency improvements

### Business Solutions
- Providing strategic recommendations for decision-makers
- Enhancing data-driven decision processes with AI-powered insights

### Educational Use
- Demonstrating AI and XAI capabilities in practical applications
- Supporting curriculum development in AI and machine learning

### Research and Development
- Advancing multi-agent explainable AI systems
- Exploring new methodologies for AI transparency and trustworthiness

## Technical Breakdown

### Modern UI Design
- Structured layout displaying user queries, responses, and explanations
- Clearly defined sections for Engineer Response, Analyst Response, and XAI Explanations

### Cutting-Edge Architecture
- **Built with Streamlit:** Ensuring quick deployment and interactive experiences
- **NLP-Powered by Hugging Face Transformers:** Delivering state-of-the-art language understanding
- **Optimized AI Model:** Utilizing `microsoft/Phi-3-mini-4k-instruct` for highly accurate and context-aware responses
- **Efficient State Management:** Using `st.session_state` to track user interactions seamlessly
- **Dynamic Response Optimization:** Customizable parameters for fine-tuned performance

### Performance Enhancements
- Optimized for both CPU and GPU to maximize efficiency
- Adaptive token length management to maintain response quality and resource efficiency

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

## Why It Matters
The Multi-Agent System with XAI Demo showcases the transformative power of AI in solving complex problems while maintaining transparency and user trust. By bridging technical precision with explainability, this system sets a new standard for intelligent automation across industries.
