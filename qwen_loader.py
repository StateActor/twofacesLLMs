import torch
 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

class ModelLoader:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
