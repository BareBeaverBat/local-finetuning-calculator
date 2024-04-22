import time

from datasets import load_dataset, Dataset

import torch


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    PreTrainedModel
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from torchinfo import summary

print("initial gpu memory usage is " + torch.cuda.memory_summary(torch.device("cuda:0")))

print("start loading dataset at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
eli5: Dataset = load_dataset("eli5_category", split="train[:1500]", trust_remote_code=True)
print("finish loading dataset at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("gpu memory usage after loading dataset is " + torch.cuda.memory_summary(torch.device("cuda:0")))

#loading dataset when running this script directly from wsl cli has wildly differing runtimes, from 1 sec to 6min to 13min, then 13min again,
# iirc in wsl-connected jupyter notebook it was taking something like 20min


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

# print("gpu memory usage before loading tokenizer is " + torch.cuda.memory_summary(torch.device("cuda:0")))
print("start loading tokenizer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("finish loading tokenizer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("gpu memory usage after loading tokenizer is " + torch.cuda.memory_summary(torch.device("cuda:0")))

fixed_len_train_dset =  ConstantLengthDataset(tokenizer, eli5_train_test["train"], "answers.text", seq_length=seq_len)
fixed_len_eval_dset =  ConstantLengthDataset(tokenizer, eli5_train_test["test"], "answers.text", seq_length=seq_len)

torch.cuda.empty_cache()
print("gpu memory usage after clearing cache is " + torch.cuda.memory_summary(torch.device("cuda:0")))

torch.cuda.memory._record_memory_history()

with torch.profiler.profile(
        with_modules=True,
         with_stack=True,
        profile_memory=True,
        record_shapes=True
                            ) as prof:
    print("start loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    model : PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    print("finish loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print("printing modules in model")
    print(model.modules().__next__())
    # for module_name in model.modules():# might revisit this if other models besides gemma 2b don't have such a nicely descriptive first model, but unlikely
    #     print(module_name)
    #print("summary of model:")
    #print(summary(model, input_size=(256128, seq_len), device=None, depth=10, col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"]))
    #todo when next try with jupyter notebook, experiment here with input_data=fixed_len_train_dset.dataset or something

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

            # todo explore the below training args, including looking at the source code
            #  (e.g. for gradient_checkpointing_enable() )
            # gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
            # gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            # Key word arguments to be passed to the `gradient_checkpointing_enable` method.
            # "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to
            # `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
            # https://pytorch.org/docs/stable/checkpoint.html

        ),
        packing=True,
        # dataset_text_field="answers.text",
        peft_config=lora_config,
        max_seq_length=seq_len
    )
    print("finish creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print("start training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    trainer.train()
    print("finish training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    pass

import os
print("cwd: " + os.getcwd())

result_files_ts_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

print("about to dump memory snapshot at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
torch.cuda.memory._dump_snapshot(f"gemma2b_mem_snapshot_{result_files_ts_str}.pickle")
print("finished dumping memory snapshot at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

print("about to dump memory timeline summary json at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
prof.export_memory_timeline(f"gemma2b_mem_timeline_summary_{result_files_ts_str}.json")
print("finished dumping memory timeline summary json and about to dump memory timeline details/raw json at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
prof.export_memory_timeline(f"gemma2b_mem_timeline_details_{result_files_ts_str}.raw.json")
print("finished dumping memory timeline details/raw json and about to dump memory timeline static visualization at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
prof.export_memory_timeline(f"gemma2b_mem_timeline_static_viz_{result_files_ts_str}.html")
print("finished dumping memory timeline static visualization about to create nicer (I think interactive) visualization of memory profiling at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

from torch.cuda._memory_viz import profile_plot
with open(f'gemma2b_mem_profile_{result_files_ts_str}.html', 'w') as f:
    f.write(profile_plot(prof))

    #when with_stack, profile_memory, and record_shapes were all True; and dataset size was 5k (leading to 10 hills in memory snapshot- 10 epochs):
    # profile_plot() ran for almost 3 hours and then was killed (by the wsl shell? by the python interpreter? not sure)

    # when I tried setting with_stack and record_shapes to False to hopefully speed it up, I got error that memory profiling requires them. oops



print("finished creating visualization of memory profiling at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


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
