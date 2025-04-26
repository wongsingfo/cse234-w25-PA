import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    # Calculate model parallel and data parallel indices
    dp_idx = rank // mp_size
    mp_idx = rank % mp_size
    
    # Create model parallel communicator (processes with same dp_idx)
    # Each model parallel group has mp_size processes
    mp_comm = comm.Split(color=mp_color, key=mp_idx)
    
    # Create data parallel communicator (processes with same mp_idx)
    # Each data parallel group has dp_size processes
    dp_color = mp_idx
    dp_comm = comm.Split(color=dp_color, key=dp_idx)
    
    # Partition dimensions based on the FC layer type
    if fc_layer in ['fc_q', 'fc_k', 'fc_v']:
        # For fc_q, fc_k, fc_v, partition along output dimension
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size
    elif fc_layer == 'fc_o':
        # For fc_o, partition along input dimension
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim
    else:
        raise ValueError(f"Unsupported FC layer type: {fc_layer}")
    
    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim

def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    
    # Use allgather to collect slices from all processes
    # This directly returns the gathered data as a list of arrays
    gathered_data = mp_comm.allgather(x)
    
    # Concatenate along the last dimension (feature dimension)
    collected_x = np.concatenate(gathered_data, axis=2)
    
    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    # Use allgather to collect slices from all processes
    # This directly returns the gathered data as a list of arrays
    gathered_data = mp_comm.allgather(out)
    
    # Concatenate along the last dimension (feature dimension)
    collected_out = np.concatenate(gathered_data, axis=2)

    return collected_out
    

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    # Get the dimensions of the input tensor
    batch_size, seq_length, out_dim = output_grad.shape
    
    # Calculate the part_out_dim for each MP node
    part_out_dim = out_dim // mp_size
    
    # Calculate the start and end indices for the slice
    start_idx = mp_group_idx * part_out_dim
    end_idx = (mp_group_idx + 1) * part_out_dim
    
    # Extract the slice corresponding to this MP node
    collected_output_grad = output_grad[:, :, start_idx:end_idx]
    
    return collected_output_grad

def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    # Get the dimensions of the local gradient
    batch_size, seq_length, in_dim = grad_x.shape
    
    # Calculate the part_in_dim for each MP node
    part_in_dim = in_dim // mp_size
    
    # Split the input tensor into chunks along the last dimension
    # Each process will own one chunk after the reduce-scatter operation
    chunks = np.zeros((mp_size, batch_size, seq_length, part_in_dim), dtype=grad_x.dtype)
    
    # Slice the input tensor into chunks for reduce-scatter
    for i in range(mp_size):
        start_idx = i * part_in_dim
        end_idx = (i + 1) * part_in_dim
        chunks[i] = grad_x[:, :, start_idx:end_idx]
    
    # Create buffer for the result
    result = np.zeros((batch_size, seq_length, part_in_dim), dtype=grad_x.dtype)
    
    # Use reduce_scatter_block to perform the reduce-scatter operation
    # This will sum corresponding chunks across processes and distribute the results
    mp_comm.Reduce_scatter(
        chunks, result, op=MPI.SUM
    )
    
    return result
    