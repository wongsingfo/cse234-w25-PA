import numpy as np

def split_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mp_size: int,
    dp_size: int,
    rank: int,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        - Data is split uniformly across data parallel (DP) groups.
        - All model parallel (MP) ranks within the same DP group share the same data.
        - The data length is guaranteed to be divisible by dp_size.
        - Do not shuffle the data indices as shuffling will be done later.
    """

    # Calculate the number of samples per DP group
    samples_per_dp = x_train.shape[0] // dp_size
    assert x_train.shape[0] % dp_size == 0, "Data length must be divisible by dp_size"

    """Examples:
         MP0     MP1
    DP0  rank 0  rank 1
    DP1  rank 2  rank 3
    DP2  rank 4  rank 5
    DP3  rank 6  rank 7
    """

    # Calculate the starting and ending indices for the DP group
    data_start_idx = (rank // mp_size) * samples_per_dp
    data_end_idx = data_start_idx + samples_per_dp

    split_x_train = x_train[data_start_idx:data_end_idx]
    split_y_train = y_train[data_start_idx:data_end_idx]

    return split_x_train, split_y_train

