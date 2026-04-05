"""
LLM-powered chatbot for explainable credit risk decisions.
Converts SHAP values + loan features into plain-language explanations.
Owner: Subramanian Raj Narayanan
"""

import os
import json
from typing import Optional
import anthropic


SYSTEM_PROMPT = """You are a credit risk analyst assistant.
Given a loan application's features and the SHAP values explaining the model's prediction,
explain in plain language (2-3 sentences) why the borrower received the predicted risk score.
Focus on the top positive and negative factors. Avoid jargon."""


def build_explanation_prompt(
    features: dict,
    shap_values: dict,
    predicted_pd: float,
) -> str:
    top_positive = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative = sorted(shap_values.items(), key=lambda x: x[1])[:5]

    prompt = (
        f"Predicted Probability of Default: {predicted_pd:.2%}\n\n"
        f"Top risk-increasing factors:\n"
        + "\n".join(f"  - {k}: SHAP={v:+.4f}, value={features.get(k, 'N/A')}" for k, v in top_positive)
        + "\n\nTop risk-reducing factors:\n"
        + "\n".join(f"  - {k}: SHAP={v:+.4f}, value={features.get(k, 'N/A')}" for k, v in top_negative)
        + "\n\nPlease explain this prediction in plain language."
    )
    return prompt


def explain_prediction(
    features: dict,
    shap_values: dict,
    predicted_pd: float,
    model: str = "claude-haiku-4-5-20251001",
    api_key: Optional[str] = None,
) -> str:
    """Call Claude to generate a plain-language explanation of one prediction."""
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
    prompt = build_explanation_prompt(features, shap_values, predicted_pd)

    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def chat_session(
    model: str = "claude-haiku-4-5-20251001",
    api_key: Optional[str] = None,
) -> None:
    """Interactive CLI chat loop for credit risk Q&A (development / demo use)."""
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
    history = []
    print("Credit Risk Chatbot — type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")
