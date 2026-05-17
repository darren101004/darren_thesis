# import time
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_path = "/media02/ltnghia24/models/DeepSeek-R1-Distill-Llama-8B"

# start_t0 = time.time()
# print(">>> [LOG] Starting load tokenizer ...")
# t0 = time.time()
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# t1 = time.time()
# print(f">>> [LOG] Tokenizer loaded in {t1 - t0:.2f}s")

# print(">>> [LOG] Starting load model ...")
# t0 = time.time()
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",       # chỉ định rõ cuda
# )
# t1 = time.time()
# print(f">>> [LOG] Model loaded in {t1 - t0:.2f}s")

# messages = [{"role": "user", "content": "What is 2+2?"}]

# print(">>> [LOG] Tokenizing prompt ...")
# t0 = time.time()
# input_ids = tokenizer.apply_chat_template(
#     messages,
#     return_tensors="pt",
#     add_generation_prompt=True
# ).to("cuda")
# t1 = time.time()
# print(f">>> [LOG] Prompt tokenized in {t1 - t0:.2f}s")

# print(">>> [LOG] Beginning generation ...")
# t0 = time.time()
# with torch.no_grad():
#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=512,
#         temperature=0.7,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,  # tránh warning
#     )
# t1 = time.time()
# print(f">>> [LOG] Generation done in {t1 - t0:.2f}s")

# print(">>> [LOG] Decoding output ...")
# response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
# print(f">>> [LOG] Output decoded")

# print("Response:", response)
# print(f">>> [LOG] TOTAL elapsed: {time.time() - start_t0:.2f}s")


from pandas import read_csv

df = read_csv("processed_data.csv")

print("Columns:", df.columns)


print("Category counts:", df["category"].value_counts())

print("Label counts:", df["label"].value_counts())

print("Text counts:", df["text"].value_counts())


sample_df = df.sample(n=2)
print("Sample dataframe:", sample_df["text"].values[1])