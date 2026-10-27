"""⑦ 生成：sarashina2.2-3b を読み込んで文章を作るだけの薄い部品。"""
import torch

from KIT_LLM_RAG.config import CFG


class Generator:
    def __init__(self, model_name: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = model_name or CFG.llm_model
        self.tok = AutoTokenizer.from_pretrained(name)
        # Colab T4 は bfloat16 非対応なので float16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="auto")
        self.model.eval()
        self.use_chat = bool(getattr(self.tok, "chat_template", None))  # instruct版なら会話形式

    @torch.no_grad()
    def __call__(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.use_chat:
            ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, return_tensors="pt")
        else:
            ids = self.tok(prompt, return_tensors="pt").input_ids
        ids = ids.to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  repetition_penalty=1.1,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
