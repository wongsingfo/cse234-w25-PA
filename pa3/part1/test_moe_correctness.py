import numpy as np
from mpi4py import MPI
from rng import get_rng, rng_context, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time
import pickle

# Correctness checksum for MoE models
correct_outputs = {
    "simple": 4.016589635077568, 
    "ep": -28.57107747979851,   
    "tp": -79.30519534272293,   
}

def sum_parameters(model):
    """
    Sum the actual parameter values in a model (not just count them)
    
    Args:
        model: MoE model instance
        
    Returns:
        Dictionary with sums of different parameter groups
    """
    result = {}
    
    if isinstance(model, SimpleMoE):
        # Sum router parameters
        router_weight_sum = np.sum(model.router.linear.weight)
        router_bias_sum = np.sum(model.router.linear.bias)
        result["router_weight_sum"] = router_weight_sum
        result["router_bias_sum"] = router_bias_sum
        
        # Sum expert parameters
        expert_fc1_weight_sum = 0
        expert_fc1_bias_sum = 0
        expert_fc2_weight_sum = 0
        expert_fc2_bias_sum = 0
        
        for expert in model.experts:
            # First layer
            expert_fc1_weight_sum += np.sum(expert.fc1.weight)
            expert_fc1_bias_sum += np.sum(expert.fc1.bias)
            # Second layer
            expert_fc2_weight_sum += np.sum(expert.fc2.weight)
            expert_fc2_bias_sum += np.sum(expert.fc2.bias)
        
        result["expert_fc1_weight_sum"] = expert_fc1_weight_sum
        result["expert_fc1_bias_sum"] = expert_fc1_bias_sum
        result["expert_fc2_weight_sum"] = expert_fc2_weight_sum
        result["expert_fc2_bias_sum"] = expert_fc2_bias_sum
    
    elif isinstance(model, MoE_EP):
        # Sum router parameters
        router_weight_sum = np.sum(model.router.linear.weight)
        router_bias_sum = np.sum(model.router.linear.bias)
        result["router_weight_sum"] = router_weight_sum
        result["router_bias_sum"] = router_bias_sum
        
        # Sum local expert parameters
        local_fc1_weight_sum = np.sum(model.expert.fc1.weight)
        local_fc1_bias_sum = np.sum(model.expert.fc1.bias)
        local_fc2_weight_sum = np.sum(model.expert.fc2.weight)
        local_fc2_bias_sum = np.sum(model.expert.fc2.bias)
        
        # Gather expert param sums from all processes
        all_fc1_weight_sums = mpi.allgather(local_fc1_weight_sum)
        all_fc1_bias_sums = mpi.allgather(local_fc1_bias_sum)
        all_fc2_weight_sums = mpi.allgather(local_fc2_weight_sum)
        all_fc2_bias_sums = mpi.allgather(local_fc2_bias_sum)
        
        result["expert_fc1_weight_sums"] = all_fc1_weight_sums
        result["expert_fc1_bias_sums"] = all_fc1_bias_sums
        result["expert_fc2_weight_sums"] = all_fc2_weight_sums
        result["expert_fc2_bias_sums"] = all_fc2_bias_sums
        result["expert_fc1_weight_sum_total"] = sum(all_fc1_weight_sums)
        result["expert_fc1_bias_sum_total"] = sum(all_fc1_bias_sums)
        result["expert_fc2_weight_sum_total"] = sum(all_fc2_weight_sums)
        result["expert_fc2_bias_sum_total"] = sum(all_fc2_bias_sums)
    
    elif isinstance(model, MoE_TP):
        # Sum router parameters
        router_weight_sum = np.sum(model.router.linear.weight)
        router_bias_sum = np.sum(model.router.linear.bias)
        result["router_weight_sum"] = router_weight_sum
        result["router_bias_sum"] = router_bias_sum
        
        # Sum expert parameters (each process has a shard of each expert)
        expert_sums = {}
        for i, expert in enumerate(model.experts):
            # First layer (sharded)
            fc1_weight_sum = np.sum(expert.fc1.weight)
            fc1_bias_sum = np.sum(expert.fc1.bias)
            # Second layer (sharded)
            fc2_weight_sum = np.sum(expert.fc2.weight)
            fc2_bias_sum = np.sum(expert.fc2.bias)
            
            expert_sums[f"expert_{i}_fc1_weight_sum_local"] = fc1_weight_sum
            expert_sums[f"expert_{i}_fc1_bias_sum_local"] = fc1_bias_sum
            expert_sums[f"expert_{i}_fc2_weight_sum_local"] = fc2_weight_sum
            expert_sums[f"expert_{i}_fc2_bias_sum_local"] = fc2_bias_sum
        
        # Gather all local sums from all processes
        all_expert_sums = mpi.allgather(expert_sums)
        
        # Combine the sharded sums for each expert
        for i in range(model.num_experts):
            fc1_weight_sum_total = sum(proc_sums[f"expert_{i}_fc1_weight_sum_local"] for proc_sums in all_expert_sums)
            fc1_bias_sum_total = sum(proc_sums[f"expert_{i}_fc1_bias_sum_local"] for proc_sums in all_expert_sums)
            fc2_weight_sum_total = sum(proc_sums[f"expert_{i}_fc2_weight_sum_local"] for proc_sums in all_expert_sums)
            fc2_bias_sum_total = sum(proc_sums[f"expert_{i}_fc2_bias_sum_local"] for proc_sums in all_expert_sums)
            
            result[f"expert_{i}_fc1_weight_sum_total"] = fc1_weight_sum_total
            result[f"expert_{i}_fc1_bias_sum_total"] = fc1_bias_sum_total
            result[f"expert_{i}_fc2_weight_sum_total"] = fc2_weight_sum_total
            result[f"expert_{i}_fc2_bias_sum_total"] = fc2_bias_sum_total
    
    return result

def print_parameter_sums(model):
    """
    Print the sums of parameter values in a model
    
    Args:
        model: MoE model instance
    """
    param_sums = sum_parameters(model)
    
    if mpi.get_rank() == 0:
        model_type = type(model).__name__
        print(f"Model type: {model_type}")
        print("Parameter sums:")
        
        for key, value in param_sums.items():
            if isinstance(value, list):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.6f}")
        
        print("-" * 40)

# Example usage
def run_moe(
    moe_type="tp", 
    batch_size=8, 
    feature_dim=32, 
    hidden_dim=128, 
    output_dim=64, 
    num_experts=None,
    topk=2,
):
    """
    Unified function to run different types of MoE models
    
    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    # Get number of experts based on MPI world size
    num_experts = mpi.get_size()

    # Generate input data
    with rng_context("testing"):
        X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        # Synchronize the input data across all processes
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)
    
    # Create appropriate MoE model
    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)
    
    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )

    print(f"Printing parameter sums for {moe_type} MoE:")
    print_parameter_sums(moe)
    
    # Run forward pass with a fixed seed.
    with rng_context("testing"):
        correctness_output = moe(X)

    return dict(
        correctness_output=correctness_output,
    )

batch_size = 4
feature_dim = 8
hidden_dim = 32
output_dim = 16
num_experts = 4
topk = 2

    
def test_simple_moe_correctness():
    # Register a different RNG for each process
    # This way, if necessary, different process will initialize different experts
    rank = mpi.get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    result = run_moe(
        "simple",
        batch_size=batch_size,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )
    correctness_output = result["correctness_output"]
    if abs(correctness_output.sum() - correct_outputs["simple"]) <= 2:
        print("Simple MoE test passed")
    else:
        print(f"Simple MoE test failed: {correctness_output.sum() = } != {correct_outputs['simple'] = }. SimpleMoE correctness checksum should be consistent.")
    return 

def test_ep_moe_correctness():
    result = run_moe("ep", batch_size=batch_size, feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_experts=num_experts, topk=topk)
    correctness_output = result["correctness_output"]
    if abs(correctness_output.sum() - correct_outputs["ep"]) <= 2:
        print("Expert Parallel MoE test passed")
    else:
        print(f"Expert Parallel MoE test failed: {correctness_output.sum() = } != {correct_outputs['ep'] = }. It is possible that some parameters are not initialized correctly, or your implementation is incorrect.")
    return

def test_tp_moe_correctness():
    result = run_moe("tp", batch_size=batch_size, feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_experts=num_experts, topk=topk)
    correctness_output = result["correctness_output"]
    if abs(correctness_output.sum() - correct_outputs["tp"]) <= 2:
        print("Tensor Parallel MoE test passed")
    else:
        print(f"Tensor Parallel MoE test failed: {correctness_output.sum() = } != {correct_outputs['tp'] = }. It is possible that some parameters are not initialized correctly, or your implementation is incorrect.")
    return

if __name__ == "__main__":
    test_simple_moe_correctness()
    test_ep_moe_correctness()
    test_tp_moe_correctness()