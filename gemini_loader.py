import os
from google import genai

GEMINI_KEY = os.getenv("GEMINI_KEY")
MODEL_NAME = "gemma-4-31b-it"

class GeminiLoader:
    def __init__(self, model_name: str = MODEL_NAME):
        if not GEMINI_KEY:
            raise EnvironmentError("Gemini API key not set.")
        self.client = genai.Client(api_key=GEMINI_KEY)
        self.model_name = model_name

    def chat_helper(self, prompt: str, system: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()



