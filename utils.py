import torch 
from torchtext.data.metrics import bleu_score
import transformers
from transformers import GPT2TokenizerFast
import random
import numpy as np


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    special_tokens = {
        "pad_token": "<|pad|>",
        "sep_token": "<|sep|>",
        "eos_token": "<|eos|>"
    }
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def gen_reply(model, tokenizer, comment, method="beam_search"):
    if len(comment) > 1024:
        comment = comment[:1024]

    max_length = len(comment) + 200
    
    if method == "beam_search":
        output = model.generate(
            comment,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=5,
            early_stopping=True 
        )
    elif method == "top_k_sampling":
        output = model.generate(
            comment,
            do_sample=True,
            max_length=max_length,
            top_k=50
        )
    elif method == "top_p_sampling":
        output = model.generate(
            comment,
            do_sample=True,
            max_length=max_length,
            top_p=0.9,
            top_k=0
        )
    else:
        output = model.generate(
            comment,
            max_length=max_length
        )

    output = tokenizer.decode(output[0])
    
    output = output.lower()
    output = output.split('<|eos|>')
    output = output[1:-1]
    
    output = list(map(lambda x: x.strip(), output))
    gen_output = output[np.random.randint(0, len(output))]
    
    while gen_output == "":
        gen_output = output[np.random.randint(0, len(output))]
    
    return gen_output
