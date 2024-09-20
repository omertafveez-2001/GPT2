from gpt2 import *
from modules import GPTConfig
import argparse


def parsing():
    parser = argparse.ArgumentParser(description='Train GPT model with different attention mechanisms')
    parser.add_argument('--mode', type=str, default='scratch', choices=['scratch', 'pretrained'],
                        help='Loading Model Mode: [scratch, pretrained]', required=True)
    parser.add_argument('--model_type', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2',
                        help='Model Type: [gpt2, gpt2-medium, gpt2-large, gpt2-xl]')
    return parser

def get_model(parser):
    args = parser.parse_args()
    
    print(f"Loading Model with the mode type: {args.mode} and model type {args.model_type}...")
    if args.mode == 'pretrained':
        model = GPT.from_pretrained(args.model_type)
    else:
        model = GPT(GPTConfig(vocab_size=50304))
    return model

