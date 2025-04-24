import argparse
import json
import math

# Reference:
# https://hao-ai-lab.github.io/cse234-w25/assets/slides/feb27.pdf
# https://github.com/meta-llama/codellama/blob/main/llama/model.py
# https://github.com/MrYxJ/calculate-flops.pytorch

def calculate_total_params_llama(model_config):
    """Calculate the total number of parameters in the model."""

    batch_size = 1 # batch size
    sequence_length = model_config['max_sequence_length'] # sequence length
    num_heads = model_config['num_attention_heads'] # number of attention heads
    hidden_size = model_config['hidden_size'] # hidden size
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    head_dim = hidden_size // num_heads # head dimension
    # round up (hidden_size * 4) * (2/3) to the nearest multiple of 256
    intermediate_size = (int((hidden_size * 4) * (2/3)) + 255) // 256 * 256
    assert intermediate_size == model_config['intermediate_size'], \
        f"intermediate_size must be equal to the model config: {intermediate_size} != {model_config['intermediate_size']}"
    vocab_size = model_config['vocab_size'] # vocab size
    num_layers = model_config['num_hidden_layers'] # number of layers

    embedding_params = vocab_size * hidden_size
    transformer_params = num_layers * (
        # RMSNorm
        hidden_size +
        # Self-attention
        num_heads * head_dim * hidden_size * 3 +
        # Output projection
        hidden_size * hidden_size +
        # RMSNorm
        hidden_size +
        # MLP: w2(F.silu(w1(x)) * w3(x))
        hidden_size * intermediate_size +
        intermediate_size * hidden_size +
        hidden_size * intermediate_size
    )
    output_params = (
        # RMSNorm
        hidden_size +
        # Linear
        vocab_size * hidden_size
    )
    
    # Total parameters
    total_params = embedding_params + transformer_params + output_params
    
    return total_params

def calculate_flops_per_token_llama(model_config, include_backward=False,
                                    sequence_length=None):
    """Calculate the FLOPs per token for the model in trillion FLOPS (TF).

    If sequence_length is not provided, use the max sequence length in the model config.
    """

    batch_size = 1 # batch size
    sequence_length = model_config['max_sequence_length'] if sequence_length is None else sequence_length
    num_heads = model_config['num_attention_heads'] # number of attention heads
    hidden_size = model_config['hidden_size'] # hidden size
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    head_dim = hidden_size // num_heads # head dimension
    intermediate_size = model_config['intermediate_size'] # intermediate size
    vocab_size = model_config['vocab_size'] # vocab size
    num_layers = model_config['num_hidden_layers'] # number of layers

    def matmul_flops(m, n, k):
        return 2 * m * n * k
    
    def rmsnorm_flops(b, s, h):
        # x^2, \sum, / ev, *gamma
        flops = 4 * b * s * h 
        # \sqrt, +eps
        flops += 2 * b * h
        return flops

    def residual_flops(b, s, h):
        # residual
        flops = b * s * h
        return flops
    
    def self_attention_flops(b, s, h, n):
        assert h % n == 0, "h must be divisible by n"
        d = h // n
        # qkv
        flops = 3 * b * matmul_flops(s, h, h)
        # RoPE
        flops += 3 * b * s * d * n
        # attention
        flops += b * matmul_flops(s, d, s) * n
        # softmax(attention): e^x, \sum, / sum
        flops += 3 * b * s * s * n
        # pv
        flops += b * matmul_flops(s, h, s)
        # output projection
        flops += b * matmul_flops(h, s, h)
        return flops
    
    def mlp_flops(b, s, h, i):
        # w1: up projection
        flops = b * matmul_flops(h, s, i)
        # activation: x * sigmoid(x)
        flops += 4 * b * s * i 
        # w2: gate
        flops += b * matmul_flops(h, s, i)
        # activation: x * silu(x)
        flops += b * s * i
        # down projection
        flops += b * matmul_flops(i, s, h)
        return flops

    embedding_flops = 0
    transformer_flops = num_layers * (
        rmsnorm_flops(batch_size, sequence_length, hidden_size) +
        self_attention_flops(batch_size, sequence_length, hidden_size, num_heads) +
        residual_flops(batch_size, sequence_length, hidden_size) +
        rmsnorm_flops(batch_size, sequence_length, hidden_size) +
        mlp_flops(batch_size, sequence_length, hidden_size, intermediate_size) +
        residual_flops(batch_size, sequence_length, hidden_size)
    )
    output_flops = (
        rmsnorm_flops(batch_size, sequence_length, hidden_size) +
        matmul_flops(hidden_size, sequence_length, vocab_size) +
        # softmax, e^x, \sum
        2 * batch_size * sequence_length * vocab_size
    )

    flops_per_token = embedding_flops + transformer_flops + output_flops
    
    if include_backward:
        flops_per_token += flops_per_token * 2
    
    return flops_per_token / 1e12 # TF

def calculate_peak_memory(model_config, total_params):
    """Calculate the peak memory usage in GB during training."""

    batch_size = 1
    num_hidden_layers = model_config['num_hidden_layers']
    hidden_size = model_config['hidden_size']
    max_sequence_length = model_config['max_sequence_length']
    
    # Model parameters (FP16)
    params_memory = total_params * 2 # ZeRO Stage 3

    # Activations (FP16, use checkpointing at each layer)
    activations_memory = batch_size * num_hidden_layers * max_sequence_length * hidden_size * 2
    
    # Optimizer states (assuming Adam: first moment, second moment, master copy of weights)
    optimizer_memory = total_params * 4 * 3 # ZeRO Stage 1
    
    # Weight gradients (FP16)
    weight_gradients_memory = total_params * 2 # ZeRO Stage 2

    # Total peak memory
    thumb_of_rule_16M = params_memory + optimizer_memory + weight_gradients_memory
    peak_memory = activations_memory + thumb_of_rule_16M
    
    return peak_memory / 1e9 # GB

def model_training_cost_analysis_llama(model_config_path):
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Calculate using the three sub-functions
    total_params = calculate_total_params_llama(model_config)
    flops_layer_TF = calculate_flops_per_token_llama(model_config,
                                                     include_backward=False)
    peak_memory_GB = calculate_peak_memory(model_config, total_params)
    
    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    #TODO you code here.
    

    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    #TODO you code here

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters/1e9:.1f}B")
        print(f"Number of TFLOPs (forward pass): {num_flops:.3f}")
        print(f"Peak memory cost: {memory_cost:.3f} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    