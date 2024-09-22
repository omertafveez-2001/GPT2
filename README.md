# GPT2
This repo is Karpathy's implementation of GPT-2. To make more sense, there is a separate markdown, `important_notes.md`, with notes on specific lines of code, although I have tried my best to put a lot of comments within files to remove any ambiguity. <br> <br>
Note that this implementation is not just from the papers of GPT and GPT-2 but also from GPT-3. This is because the paper on GPT-3 has richer details on training and its configurations.<br>

# Attached below is a closeup for each important script in this repository.

### dataloader.py
This file consists of the class `Dataloader,` which loads the text file into tokens in batches. It also shards data if DDP is being used. 

### gpt2.py
This is the master file consisting of the `GPT` class. In this file, a pre-trained GPT-2 is loaded if the `pretrained` is set to True, otherwise the pre-trained GPT-2 is loaded with random initialization for training GPT-2 from scratch. It's worth noting that the original GPT-2 is coded in `TensorFlow`, which is why additional lines of code are used to load in keys that are well configured with `Pytorch` because no one uses TensorFlow anymore.

### input.txt
This is a smaller dataset, `Shakespearean Text.`

### parser.py
This is an additional setup for cleaner experimentation. This consists of the `argparse` library, used for arguments within the command line. So, using this file, you can pass arguments within the command line to ensure those configurations are loaded for training. 

### modules.py
Another master file has all the modules of a transformer, including Vision Transformer, FeedForward Network, PatchEmbeddings, etc.

### train_gpt2.py
As the name suggests, this is the master file for training functions and training loop. This is the file that will be run over the command line to train GPT-2.

