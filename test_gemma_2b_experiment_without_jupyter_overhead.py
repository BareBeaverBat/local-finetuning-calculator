import time

from datasets import load_dataset, Dataset

import torch


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

print("initial gpu memory usage is " + torch.cuda.memory_summary(torch.device("cuda:0")))

print("start loading dataset at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
eli5: Dataset = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
print("finish loading dataset at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("gpu memory usage after loading dataset is " + torch.cuda.memory_summary(torch.device("cuda:0")))


eli5_train_test = eli5.train_test_split(test_size=0.2)

model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,#todo explore whether it actually increases overall memory usage to change this to float32
    bnb_4bit_use_double_quant=True# reminder that this makes computing the total memory used by the frozen weights even more complicated, something about reducing the size of the quantization constants that are used for remembering how to dequantize a given block of quantized weights? by the equivalent of 0.4 bits per parameter, per docs
)


lora_config = LoraConfig(
    r=4,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", # just lora'ing attention heads for now, to mimic original LoRA paper
                    #"gate_proj", "up_proj", "down_proj" todo what is gate_proj? I think up_proj is first weights matrix of MLP block (fan out) and down_proj is second weights matrix of MLP block (fan in), but no idea what gate_proj is
                    ],# reminder, can use "all-linear" (not inside list) for the expansive case https://huggingface.co/docs/peft/developer_guides/lora#qlora-style-training
    task_type="CAUSAL_LM",
    use_rslora=True
    #todo investigate whether it's worth trying Dora, iirc that was said to be especially helpful when lora rank is low
)

seq_len = 128

eli5_train_test = eli5_train_test.flatten()

def fix_data(record):
    '''
    make dataset usable by TRL (i.e. its classes have a dataset_text_field param, and that column must be string-type,
    not list<string> type)
    :param record: record where the text column is actually a length-1 list column
    :return: record where the text column is straightforwardly a text-type column
    '''
    record["answers.text"] = record["answers.text"][0]
    return record

eli5_train_test = eli5_train_test.map(fix_data)

print("gpu memory usage before loading tokenizer is " + torch.cuda.memory_summary(torch.device("cuda:0")))
print("start loading tokenizer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("finish loading tokenizer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("gpu memory usage after loading tokenizer is " + torch.cuda.memory_summary(torch.device("cuda:0")))

fixed_len_train_dset =  ConstantLengthDataset(tokenizer, eli5_train_test["train"], "answers.text", seq_length=seq_len)
fixed_len_eval_dset =  ConstantLengthDataset(tokenizer, eli5_train_test["test"], "answers.text", seq_length=seq_len)

torch.cuda.empty_cache()
print("gpu memory usage after clearing cache is " + torch.cuda.memory_summary(torch.device("cuda:0")))


with torch.profiler.profile(with_stack=True,
                            profile_memory=True,
                            record_shapes=True) as prof:
    print("start loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    print("finish loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("start creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    trainer = SFTTrainer(
        model=model,
        train_dataset=fixed_len_train_dset,
        eval_dataset=fixed_len_eval_dset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            # gradient_accumulation_steps=4,#todo don't want to touch this until I understand it
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_32bit"#can try paged_adamw_8bit in absolute worst case
        ),
        packing=True,
        # dataset_text_field="answers.text",
        peft_config=lora_config
    )
    print("finish creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print("start training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    trainer.train()
    print("finish training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    pass


from torch.cuda._memory_viz import profile_plot
with open('/mnt/c/Users/ssili/PycharmProjects/local-finetuning-calculator/gemma2b_1st_mem_profile.html', 'w') as f:
    f.write(profile_plot(prof))

# print("gpu memory usage before loading model is " + torch.cuda.memory_summary(torch.device("cuda:0")))
# print("start loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
# print("finish loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#
# print("gpu memory usage after loading model is " + torch.cuda.memory_summary(torch.device("cuda:0")))

# print("gpu memory usage before create trainer is " + torch.cuda.memory_summary(torch.device("cuda:0")))
# print("start creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=fixed_len_train_dset,
#     eval_dataset=fixed_len_eval_dset,
#     args=TrainingArguments(
#         per_device_train_batch_size=1,
#         # gradient_accumulation_steps=4,#todo don't want to touch this until I understand it
#         warmup_steps=2,
#         max_steps=10,
#         learning_rate=2e-4,
#         fp16=True,
#         logging_steps=1,
#         output_dir="outputs",
#         optim="paged_adamw_32bit"#can try paged_adamw_8bit in absolute worst case
#     ),
#     packing=True,
#     # dataset_text_field="answers.text",
#     peft_config=lora_config
# )
# print("finish creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("gpu memory usage after creating trainer is " + torch.cuda.memory_summary(torch.device("cuda:0")))
# torch.cuda.empty_cache()
# print("gpu memory usage after clearing cache is " + torch.cuda.memory_summary(torch.device("cuda:0")))

# torch.cuda.memory._record_memory_history()

# print("gpu memory usage before start training is " + torch.cuda.memory_summary(torch.device("cuda:0")))
# print("start training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# trainer.train()
# print("finish training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("gpu memory usage after training is " + torch.cuda.memory_summary(torch.device("cuda:0")))

# torch.cuda.memory._dump_snapshot("gemma2b_3rd_mem_snapshot.pickle")
