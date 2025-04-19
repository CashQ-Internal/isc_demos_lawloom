### INFO: This is a helper script to allow participants to confirm their model is working!
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from fsdp_utils import AppState

adapter_name = "ExampleLora"

# INFO: This is a helper to map model names to StrongCompute Dataset ID's which store their weights!
model_weight_ids = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "38b32289-7d34-4c72-9546-9d480f676840",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "a792646c-39f5-4971-a169-425324fec87b",
}

# TODO: set this to the model you chose from the dropdown at container startup!
MODEL_NAME_SETME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# mounted_dataset_path = f"/data/{model_weight_ids[MODEL_NAME_SETME]}"
mounted_dataset_path = MODEL_NAME_SETME

# INFO: Loads the model WEIGHTS (assuming you've mounted it to your container!)
tokenizer = AutoTokenizer.from_pretrained(mounted_dataset_path)
model = AutoModelForCausalLM.from_pretrained(
    mounted_dataset_path,
    use_cache=False,
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0, # set to zero to see identical loss on all ranks
)

model = LoraModel(model, lora_config, adapter_name).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
state_dict = { "app": AppState(model, optimizer)}
# dcp.load(state_dict=state_dict, checkpoint_id="/shared/artifacts/<experiment-id>/checkpoints/CHKxx") ## UPDATE WITH PATH TO CHECKPOINT DIRECTORY
dcp.load(state_dict=state_dict, checkpoint_id="/shared/artifacts/5252730d-213f-4038-8bfb-ad21f594e4ed/checkpoints/AtomicDirectory_checkpoint_69")

prompt = "Where in the Federal Rules of Civil Procedure are notice requirements described?"
prompt="""
You are LawLoom Compliance Assistant. You must answer strictly with a single letter: A, B, C, or D. Do not provide explanations, reasoning, or any additional text. If uncertain, choose the most likely letter. Respond only with the letter.\n\nQuestion 1:\nA small community bank observes a business customer making daily cash deposits of $9,999, alternating between two nearby branches. When interviewed, the customer provides business invoices that nominally support the transactions. The Bank Secrecy Act requires a SAR to be filed if:\n* A. The total amount exceeds $10,000 over a rolling 30-day period\n* B. The branch manager has definitively determined the customer acted with criminal intent\n* C. The pattern indicates structuring or attempt to evade reporting requirements, regardless of apparent business justification\n* D. The customer declines to provide additional beneficial ownership information\n\nQuestion 2:\nWhich of the following customers is exempt from the requirement to identify their beneficial owners per FinCEN'\''s CDD Final Rule?\n* A. A U.S. public company listed on the New York Stock Exchange\n* B. An SEC-registered investment advisor\n* C. A foreign financial institution based in the EU and supervised by its local AML authority\n* D. A pooled investment vehicle established under Cayman law with U.S. investors\n\nQuestion 3:\nCompany Alpha is 30% owned by SDN #1 and 25% owned by Company Beta. Company Beta is 80% owned by SDN #2. Under OFAC’s 50 Percent Rule, is Company Alpha blocked?\n* A. Yes, because Company Alpha is directly or indirectly majority-owned by SDNs\n* B. No, because no single SDN owns more than 50%\n* C. No, because indirect ownership through another company does not trigger blocking\n* D. Yes, because SDN #1 and SDN #2'\''s interests must be aggregated over Company Alpha
"""

# https://arxiv.org/abs/2501.12948
deepseek_r1_input = f'''
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {prompt}. Assistant:'''

encoding = tokenizer(deepseek_r1_input, return_tensors="pt")

input_ids = encoding['input_ids'].to("cuda")
attention_mask = encoding['attention_mask'].to("cuda")

generate_ids = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500, do_sample=True, temperature=0.3)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(answer[0])
