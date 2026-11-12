import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

model_id = "sbintuitions/sarashina2.2-3b"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16          # torch_dtype → dtype
).to(device)

# ↓ ここから下を置き換え
prompt = "質問: 金沢工業大学はどこにありますか?\n回答:"
inputs = tok(prompt, return_tensors="pt").to(device)

t = time.time()
out = model.generate(
    **inputs,
    max_new_tokens=100,
    pad_token_id=tok.eos_token_id,
    stop_strings=["\n"],   # 改行が出たら生成を止める
    tokenizer=tok,         # stop_strings を使うときに必要
)
gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
answer = gen.split("\n")[0].strip()   # 最初の1行だけ使う

print("回答:", answer)
print(f"生成時間: {time.time() - t:.1f} 秒")