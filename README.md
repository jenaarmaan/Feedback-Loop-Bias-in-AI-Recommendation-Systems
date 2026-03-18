# 🧠 Feedback-Loop Bias Simulator (AI Challenge Day 19/50)

This project simulates how AI recommendation systems can create self-reinforcing loops, leading to popularity bias and reduced content diversity.

## 🧠 Core Problem Statement
AI systems learn from their own outputs, reinforcing bias.
>"Recommendation systems amplify popularity and suppress diversity over time."

## 🎯 Features
1. **Recommendation Simulation**: Simulates a Top-K recommender system.
2. **User Interaction Simulation**: Models user clicks based on both intrinsic quality and exposure (popularity bias).
3. **Exposure Imbalance Metric**: Tracks bias using the Gini Coefficient.
4. **Mitigation Experiment**: Epsilon-Greedy exploration to mitigate the feedback loop.
5. **Bias Growth Visualization**: Live dashboard using Streamlit to track metrics over time.

## 🚀 How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🚀 Deployment (Hugging Face Spaces)
This application is designed for deployment on Hugging Face Spaces using the Streamlit SDK.

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Push the repository files (`app.py`, `simulator.py`, `requirements.txt`).
