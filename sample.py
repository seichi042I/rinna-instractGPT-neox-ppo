prompt = [
    {
        "speaker": "ユーザー",
        "text": "コンタクトレンズを慣れるにはどうすればよいですか？"
    },
    {
        "speaker": "システム",
        "text": "これについて具体的に説明していただけますか？何が難しいのでしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": "目が痛いのです。"
    },
    {
        "speaker": "システム",
        "text": "分かりました、コンタクトレンズをつけると目がかゆくなるということですね。思った以上にレンズを外す必要があるでしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": "いえ、レンズは外しませんが、目が赤くなるんです。"
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]
prompt = "<NL>".join(prompt)
prompt = (
    prompt
    + "<NL>"
    + "システム: "
)
print(prompt)
# "ユーザー: コンタクトレンズを慣れるにはどうすればよいですか？<NL>システム: これについて具体的に説明していただけますか？何が難しいのでしょうか？<NL>ユーザー: 目が痛いのです。<NL>システム: 分かりました、コンタクトレンズをつけると目がかゆくなるということですね。思った以上にレンズを外す必要があるでしょうか？<NL>ユーザー: いえ、レンズは外しませんが、目が赤くなるんです。<NL>システム: "

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo")

if torch.cuda.is_available():
    model = model.to("cuda")

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=128,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
output = output.replace("<NL>", "\n")
print(output)
"""それは、コンタクトレンズが目に合わないために起こることがあります。レンズが目の表面に長時間触れ続けることが原因となることがあります。また、コンタクトレンズが汚れている可能性もあります。コンタクトレンズケースを定期的に洗浄したり、コンタクトレンズを正しくフィットさせるようにしたりすることが役立ちます。</s>"""
