# local-finetuning-calculator
final project for OSU CSE6431

Report on project findings can be found [here](vram_predictor_report.pdf) 

## General setup
1. Install cuda from nvidia website (neither tool nor experiments will work for non-Nvidia GPU's, or if your computer doesn't have CUDA set up)

## Additional setup for predictor
1. Ensure Python 3.10 or later is installed
2. Create a virtual environment, and install the required packages there (torch >=2.2.2, marshmallow, marshmallow_dataclass)
3. Run the predictor with the following command to see a full list of options:
```
python vram_use_predictor.py -h
```
4. Run the tool to predict outcomes for a wide range of configurations of one model and report the best configurations 
Note that current working directory must be at the same level as the script and the model_details folder:
```
python vram_use_predictor.py google/gemma_2b.json
```
5. Run the tool to predict outcomes for a constrained set of configurations for one model:
```
python vram_use_predictor.py google/gemma_7b.json --lora-mlp True
```
or 
```
python vram_use_predictor.py google/gemma_7b.json --lora-embed False --batch-size 2 --num-configs 50
```

## Additional setup for replicating the experiments
1. Install WSL (Windows Subsystem for Linux) if you are using Windows
2. Inside the Linux environment, create a Python virtual environment and install the required packages (e.g. torch >=2.2.2, bitsandbytes, trl, peft, transformers)
3. Execute the training runs inside the Linux environment,   
3a. Either from command line (using one of the "test_gemma_?b_experiment_from_cli.py" scripts, customized as necessary to test a particular scenario)   
3b. or by setting up Pycharm with a remote interpreter inside the Linux environment and then running the jupyter notebooks from Pycharm  
3b-i. if using WSL, you may need to uninstall/reinstall Pycharm for it to notice WSL as a possible source of remote interpreters  