# **IMPORTANT NOTES**


- NOTE1 : The GPT-2 model is a causal language model, which means that it is an autoregressive model that conditions on the past tokens to predict the next token.

- NOTE2: When initializing the model with the pretrained weights, the model returns better outputs compared to a random model initialization from pytorch. 

- NOTE3: When inferencing the model, a loop is run until max_length is reached. The loop iterates for all the batches to get the next token in each iteration.
This is done by getting the top-k logits from the probability distribution after applying Softmax. Then, a token is sampled and appended to the sequence.

- NOTE4: We should be seeing gains in the training and less overfitting this is because a lot of tokens in the 50k vocab size do not occur in our dataset.
Therefore, it makes sense to drag the biases down to infinity. This is done by masking the logits of the tokens that are not in the dataset. 
On 50 epochs it will not come down to 0 loss.

- NOTE5: The weights of the text embeddings and the linear layer (lmhead) are shared which is why we see that the weights are the same in the model and they point
to the same memory. This is because the weights are shared between the embeddings and the linear layer. This is because similar tokens should have similar embeddings.

- NOTE6: FP16 has shorter dynamic range, while BF16 has a longer dynamic range. FP16 is faster but less accurate. BF16 is slower but more accurate.

- NOTE7: Torch.compile is a new feature in PyTorch that allows you to compile a model to TorchScript and optimize it for a specific device. It makes the code run faster
but the compilation time a little slower. The speed delay usually comes from the python overhead and GPU rewrites.
torch.compile removes the python interpreter overhead to torchscript. 

- NOTE8: Flashattention is a kernal fusion algorithm that fuses the attention. It does more flops than standard attention but is much more fasters since its very mindful
of its memory. It is a very efficient way of doing attention.
It relies on online softmax trick: you can incremently evaluate softmax without realizing all the inputs. 

- NOTE9: Ugly number are numbers used in the transformers models that are not a power of 2 since most of the kernels and cuda itself is written in a power of 2.
Therefore the vocab size is 50257 which is an ugly number. Vocab size is easier to fix by increasing to the earliest number that is a power of 2.

- NOTE10: Gradient clipping was used in the GPT-3 but have been replicated in gpt-2 as well in this repo. Gradient Clipping helps in setting a maximum norm. It prevents the model from getting to big shocks in terms of the gradient magnitude. So it stabilizes the training!

- NOTE11: Batch size schedular gradually increases the batch size linearly but complicates the computational resource required. 

- NOTE12: Weight decay is a sort of regularization on weights that are either not biases nor 1D tensors such as LayerNorm.

- NOTE13: FusedAdamW is the fused kernel for adamw operations.

- NOTE14: We use 0.5M batch size of simulation to literally avoid `gpu explosion` since our gpus cannot handle a batch size of roughly 500. We do this by gradient accumulation. Gradient Accumulation allows us to simulate any arbritrary batch size we set. We just need to run them for longer and accumulate gradients from the tokens.

- NOTE15: DDP
 - ddp_rank: # of GPU 
 - ddp_local_rank: # of GPU on local
 - DDP takes an average of all the gpus across backward pass and then synchronizes the gradients.

 - NOTE16: `loss_acum += loss.detach()` is outside the dpp containter therefore when we print the loss, we only get the loss values of the master processes (rank 0) however we want to the avergae loss across all the ranks.



