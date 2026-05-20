import torch
 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from google import genai

GOOGLEAI_KEY = os.getenv("GOOGLEAI_KEY")

class QwenLoader:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
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
    def __init__(self):
        if not GOOGLEAI_KEY:
            raise EnvironmentError("Gemma API key not set.")
        self.client = genai.Client(api_key=GOOGLEAI_KEY)
        self.model_name = "gemini-3.1-flash-lite"#"gemma-4-31b-it"

    def helper_chat(self, prompt: str, system: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()
