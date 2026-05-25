import torch
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from google import genai
from google.genai import types as genai_types
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GOOGLEAI_KEY = os.getenv("GOOGLEAI_KEY")

class QwenLoader:
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Loading local model: {model_name}")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
    
    #claude assisted
    def chat(
        self,
        history: list[dict],
        max_new_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> str:
        prompt = (
            self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            if self.tokenizer.chat_template
            else "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history) + "\n\nAssistant:"
        )
        out = self._pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=top_p,
                         repetition_penalty=repetition_penalty, return_full_text=False)
        return out[0]["generated_text"].strip()
 
    def chat_single(self, prompt: str, system: str | None = None) -> str:
        history = ([{"role": "system", "content": system}] if system else [])
        history.append({"role": "user", "content": prompt})
        return self.chat(history)
    
class GemmaLoader:
    DEFAULT_MODEL = "gemini-3.1-flash-lite"#"gemma-4-31b-it"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not GOOGLEAI_KEY:
            raise EnvironmentError("Gemma API key not set.")
        self.client = genai.Client(api_key=GOOGLEAI_KEY)
        self.model_name = model_name

    def helper_chat(self, prompt: str, system: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()

class GroqLoader:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("Groq API key not set (GROQ_API_KEY).")
        self.client = Groq(api_key=key)
        self.model_name = model_name

    def chat(
        self,
        history: list[dict],
        max_new_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        retries: int = 5,
    ) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s...
                    print(f"[WARN] Rate limited, waiting {wait}s (attempt {attempt + 1}/{retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Groq rate limit exceeded after {retries} retries")

    def chat_single(self, prompt: str, system: str | None = None) -> str:
        history = ([{"role": "system", "content": system}] if system else [])
        history.append({"role": "user", "content": prompt})
        return self.chat(history)

    def helper_chat(self, prompt: str, system: str | None = None) -> str:
        return self.chat_single(prompt, system)

class GeminiJudge:
    DEFAULT_MODEL = "models/gemma-4-31b-it"

    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_KEY")
        if not key:
            raise EnvironmentError("Gemini API key not set.")
        self.client = genai.Client(api_key=key)
        self.model_name = model_name
        self._types = genai_types

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(temperature=0),
        )
        return response.text.strip()

class GroqJudge:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("Groq API key not set (GROQ_API_KEY).")
        self.client = Groq(api_key=key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=64,
        )
        return response.choices[0].message.content.strip()


class LocalJudge:
    DEFAULT_MODEL = "google/gemma-2-2b-it"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Loading local model: {model_name}")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self.pipe(formatted)
        return output[0]["generated_text"].strip()
