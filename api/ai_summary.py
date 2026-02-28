import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_portfolio_summary(data):
    prompt = f"""
    You are a professional financial analyst.

    Analyze the following portfolio metrics:

    {data}

    Provide:
    - Risk interpretation
    - Return quality
    - Diversification comments
    - Optimization insight
    - Brief improvement suggestion

    Keep it concise and professional.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini Error:", e)
        return "AI summary currently unavailable."