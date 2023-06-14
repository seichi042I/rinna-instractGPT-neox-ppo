# 基本パラメータ
model_name = "rinna/japanese-gpt-neox-3.6b"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "lora-rinna-3.6b"
output_dir = "lora-rinna-3.6b-results"

#------------------------------------------------------------------------
from transformers import AutoTokenizer

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# スペシャルトークンの確認
print("\n\n<special token check>")
print(tokenizer.special_tokens_map)
print("bos_token :", tokenizer.bos_token, ",", tokenizer.bos_token_id)
print("eos_token :", tokenizer.eos_token, ",", tokenizer.eos_token_id)
print("unk_token :", tokenizer.unk_token, ",", tokenizer.unk_token_id)
print("pad_token :", tokenizer.pad_token, ",", tokenizer.pad_token_id)

CUTOFF_LEN = 256  # コンテキスト長

# トークナイズ
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }

# トークナイズの動作確認
print("\n\n<tokenize function check>")
print(tokenize("hi there", tokenizer))

#------------------------------------------------------------------------
from datasets import load_dataset

# データセットの準備
data = load_dataset(dataset)

# データセットの確認
print("\n\n<dataset check>")
print(data["train"][5])

#------------------------------------------------------------------------
# プロンプトテンプレートの準備
def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""### 指示:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 回答:
{data_point["output"]}"""
    else:
        result = f"""### 指示:
{data_point["instruction"]}

### 回答:
{data_point["output"]}"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result

# プロンプトテンプレートの確認
print("\n\n<prompt template check>")
print(generate_prompt(data["train"][5]))

#------------------------------------------------------------------------
VAL_SET_SIZE = 2000

# 学習データと検証データの準備
train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

from transformers import AutoModelForCausalLM
# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map='auto',
)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# LoRAのパラメータ
lora_config = LoraConfig(
    r= 8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# モデルの前処理
model = prepare_model_for_int8_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

#------------------------------------------------------------------------
import transformers
eval_steps = 200
save_steps = 200
logging_steps = 20

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to="none",
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 学習の実行
model.config.use_cache = False
trainer.train() 
model.config.use_cache = True

# LoRAモデルの保存
trainer.model.save_pretrained(peft_name)