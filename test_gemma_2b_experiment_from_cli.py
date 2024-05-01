#%%
from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
#%%
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    PreTrainedModel
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

# from torchinfo import summary

#%%
import os
import time
#%%
# notebook_login()#Grader- you may need to uncomment this the first time you run this script and then comment it out for further runs
#%%
eli5: Dataset = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
#%%
eli5_train_test = eli5.train_test_split(test_size=0.2)
#%%
# eli5_train_test["train"][0]
#%%
model_id = "google/gemma-2b"
four_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",#todo try 8 bit quantization (only int8 available in bnb) for gemma _2_ b
    bnb_4bit_compute_dtype=torch.float16,#todo explore whether it actually increases overall memory usage to change this to float32
    bnb_4bit_use_double_quant=True# reminder that this makes computing the total memory used by the frozen weights even more complicated, something about reducing the size of the quantization constants that are used for remembering how to dequantize a given block of quantized weights? by the equivalent of 0.4 bits per parameter, per docs
)

eight_bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

bnb_config = four_bnb_config



lora_config = LoraConfig(
    r=2,
    target_modules=[#"embed_tokens",
        # "q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"
                    ],# reminder, can use "all-linear" (not inside list) for the expansive case https://huggingface.co/docs/peft/developer_guides/lora#qlora-style-training
    task_type="CAUSAL_LM",
    use_rslora=True
    #todo investigate whether it's worth trying Dora, iirc that was said to be especially helpful when lora rank is low
)

seq_len = 4

#%%

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
#%%
tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.padding_side = "right" #todo based on runtime warning while building SFTTrainer, can try this if problems occur, but by default I think I should respect the default for the Gemma Tokenizer

fixed_len_train_dset =  ConstantLengthDataset(tokenizer, eli5_train_test["train"], "answers.text", seq_length=seq_len)
fixed_len_eval_dset =  ConstantLengthDataset(tokenizer, eli5_train_test["test"], "answers.text", seq_length=seq_len)

#%%
torch.cuda.memory.get_allocator_backend()
#%%
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
#the above completely breaks the visualization (somehow the graph seems to say that tens of gigabytes of ram are being used, but I'm running it on just 8gb vram gpu)
#%%
#try messing with PYTORCH_CUDA_ALLOC_CONF to make memory profiles more understandable
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
#?try using cudaMallocAsync instead of pytorch native for backend
# couldn't get this to work; I suspect some problem with setting environment variables in a jupyter notebook from windows but the notebook (and the pytorch code) is running in wsl


#if cudamallocasync doesn't make things clearer, switch back to pytorch native backend but mess with roundup_power2_divisions option
# maybe try [128:32,256:32,512:64,1024:128,2048:256] so hopefuly things get broken down at 4mib granularity
# tried this via torch.cuda.memory._set_allocator_settings, but it only changed the 124mib blocks at peak to 126mib blocks, idk why/how

#%%
# torch.cuda.memory._set_allocator_settings("roundup_power2_divisions:[128:32,256:32,512:64,1024:128,2048:256]")

#%%
torch.cuda.memory.get_allocator_backend()
#%%
torch.cuda.empty_cache()
print("gpu memory usage after clearing cache is " + torch.cuda.memory_summary(torch.device("cuda:0")))
#%%
torch.cuda.memory._record_memory_history()
#%%
# print("start loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# test_model : PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
# print("finish loading model at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#%%
# print("printing modules in model")
# print(test_model.modules().__next__())
#%%
# lora_model = PeftModel(test_model, lora_config)
#%%
# print("printing modules in lora model")
# print(lora_model.modules().__next__())
#%%

#%%

#%%

#%%
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
    # if try torchinfo again, experiment here with input_data=fixed_len_train_dset.dataset or something

    print("start creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    trainer = SFTTrainer(
        model=model,
        train_dataset=fixed_len_train_dset,
        eval_dataset=fixed_len_eval_dset,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            # gradient_accumulation_steps=4,#todo don't want to touch this until I understand it
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            # fp16=True,#never turn this on! it adds a 1000mib chunk to the very peak of memory usage in the middle of each epoch!
            logging_steps=1,
            output_dir="ml_outputs",
            optim="paged_adamw_32bit",#can try paged_adamw_8bit in absolute worst case
            gradient_checkpointing=True
            # Apparently pytorch vram viz really, really does not like quantization; before turning on grad checkpointing, it said
            # the frozen/quantized weights were "unknown"; now, it says they're "inputs" lmao
            # Regardless, this does substantially reduce the height of the peaks (by ~450mib in scenario4) and sometimes moves them to the end of the epoch instead of the middle
            #
        ),
        packing=True,#tried disabling this (scenario 12), didn't seem to meaningfully change the shape of the memory visualization over time
        # dataset_text_field="answers.text",
        peft_config=lora_config,
        max_seq_length=seq_len
    )
    print("finish creating trainer at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print("start training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    trainer.train()
    print("finish training at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

#%%
# torch.cuda.memory.get_allocator_backend()
#%%
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]
#%%
os.getcwd()
#%%
result_files_ts_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

print("about to dump memory snapshot at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
torch.cuda.memory._dump_snapshot(f"gemma2b_mem_snapshot_{result_files_ts_str}.pickle")
print("finished dumping memory snapshot at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#%%
# print("about to dump memory timeline summary json at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# prof.export_memory_timeline(f"gemma2b_mem_timeline_summary_{result_files_ts_str}.json")
# print("finished dumping memory timeline summary json and about to dump memory timeline details/raw json at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# prof.export_memory_timeline(f"gemma2b_mem_timeline_details_{result_files_ts_str}.raw.json")
# print("finished dumping memory timeline details/raw json at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("about to dump memory timeline static visualization at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# prof.export_memory_timeline(f"gemma2b_mem_timeline_static_viz_{result_files_ts_str}.html")
# print("finished dumping memory timeline static visualization about to create nicer (I think interactive) visualization of memory profiling at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#%%
print("about to start creating visualization of memory profiling at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

from torch.cuda._memory_viz import profile_plot
with open(f'gemma2b_mem_profile_{result_files_ts_str}.html', 'w') as f:
    f.write(profile_plot(prof))

    #when with_stack, profile_memory, and record_shapes were all True; and dataset size was 5k (leading to 10 hills in memory snapshot- 10 epochs):
    # profile_plot() ran for almost 3 hours and then was killed (by the wsl shell? by the python interpreter? not sure)
    # when I tried setting with_stack and record_shapes to False to hopefully speed it up, I got error that memory profiling requires them. oops
    # the above was all due to laptop only having 16gb of main ram, afaict. profile_plot works fine (if moderately slower than earlier profiler file creation steps) on desktop with 64gb of ram
print("finished creating visualization of memory profiling at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#%%
torch.cuda.memory._record_memory_history(enabled=None)
#%%

#%%
