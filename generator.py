"""⑦ 生成：sarashina2.2-3b を読み込んで文章を作るだけの薄い部品。

実行環境に合わせて自動で切り替える：
  NVIDIA GPU（Colab T4 など）→ cuda / float16（bf16対応GPUなら bfloat16）
  Mac（Apple Silicon）       → mps  / float16
  それ以外                   → cpu  / float32
"""
import torch

from config import CFG


def pick_device():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class Generator:
    def __init__(self, model_name: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = model_name or CFG.llm_model
        self.device, dtype = pick_device()
        print(f"generator: {name} on {self.device} ({dtype})")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.use_chat = bool(getattr(self.tok, "chat_template", None))  # instruct版なら会話形式

    @torch.no_grad()
    def __call__(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.use_chat:
            ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, return_tensors="pt")
        else:
            ids = self.tok(prompt, return_tensors="pt").input_ids
        ids = ids.to(self.device)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=max_new_tokens, do_sample=False,
                                  repetition_penalty=1.1,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)