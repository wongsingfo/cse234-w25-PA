import functools
from typing import Callable, Tuple, List, Optional, Dict

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def linear(X: ad.Node, W: ad.Node, b: Optional[ad.Node], 
           batch_size: int, seq_length: int, in_dim: int, out_dim: int) -> ad.Node:
    """Construct the computational graph for a linear layer.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, in_dim), denoting the input data.
    W: ad.Node
        A node in shape (in_dim, out_dim), denoting the weight matrix.
    b: Optional[ad.Node]
        A node in shape (out_dim,), denoting the bias vector.
    """
    W_ = ad.broadcast(W,
                      input_shape=(in_dim, out_dim),
                      target_shape=(batch_size, in_dim, out_dim))
    XW = ad.matmul(X, W_) # (batch_size, seq_length, out_dim)
    if b is not None:
        b_ = ad.broadcast(b,
                          input_shape=(out_dim,),
                          target_shape=(batch_size, seq_length, out_dim))
        XW = XW + b_
    return XW


def single_head_attention(X: ad.Node, WQ: ad.Node, WK: ad.Node, WV: ad.Node, WO: ad.Node,
                          batch_size: int, seq_length: int, in_dim: int, hidden_dim: int, out_dim: int,
                          rename: bool = True) -> ad.Node:
    """Construct the computational graph for a single head attention layer.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, in_dim), denoting the input data.
    WQ: ad.Node
        A node in shape (in_dim, hidden_dim), denoting the query matrix.
    WK: ad.Node
        A node in shape (in_dim, hidden_dim), denoting the key matrix.
    WV: ad.Node
        A node in shape (in_dim, hidden_dim), denoting the value matrix.
    WO: ad.Node
        A node in shape (hidden_dim, out_dim), denoting the output matrix.
    """
    Q = linear(X, WQ, None, batch_size, seq_length, in_dim, hidden_dim) # (batch_size, seq_length, hidden_dim)
    K = linear(X, WK, None, batch_size, seq_length, in_dim, hidden_dim) # (batch_size, seq_length, hidden_dim)
    V = linear(X, WV, None, batch_size, seq_length, in_dim, hidden_dim) # (batch_size, seq_length, hidden_dim)
    QK = ad.matmul(Q, ad.transpose(K, 1, 2)) / np.sqrt(hidden_dim) # (batch_size, seq_length, seq_length)
    QK_softmax = ad.softmax(QK) # (batch_size, seq_length, seq_length)
    QKV = ad.matmul(QK_softmax, V) # (batch_size, seq_length, hidden_dim)
    O = linear(QKV, WO, None, batch_size, seq_length, hidden_dim, out_dim) # (batch_size, seq_length, out_dim)
    if rename:
        Q.name = "Q"
        K.name = "K"
        V.name = "V"
        QK.name = "QK"
        QK_softmax.name = "Attention"
        QKV.name = "QKV"
        O.name = "O"

    return O


def transformer(X: ad.Node, nodes: Dict[str, ad.Node], 
                input_dim: int, model_dim: int, seq_length: int, eps: float, batch_size: int, num_classes: int,
                rename: bool = True) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: Dict[str, ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    X1 = single_head_attention(X, nodes['WQ'], nodes['WK'], nodes['WV'], nodes['WO'],
                               batch_size, seq_length, input_dim, model_dim, model_dim)
    # (batch_size, seq_length, model_dim)

    X2 = ad.layernorm(X1, normalized_shape=(model_dim,), eps=eps)
    # (batch_size, seq_length, model_dim)
    X21 = linear(X2, nodes['W1'], nodes['b1'], batch_size, seq_length, model_dim, model_dim)

    X3 = ad.relu(X21)

    X4 = ad.layernorm(X3, normalized_shape=(model_dim,), eps=eps)

    X5 = linear(X4, nodes['W2'], nodes['b2'], batch_size, seq_length, model_dim, num_classes) 
    # (batch_size, seq_length, num_classes)

    X6 = ad.mean(X5, dim=(1,), keepdim=False)
    # (batch_size, num_classes)

    if rename:
        X1.name = "Transformer"
        X2.name = "LayerNorm1"
        X3.name = "ReLU"
        X4.name = "LayerNorm2"
        X5.name = "Linear"
        X6.name = "Mean"

    return X6
    
def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """

    # Question: Why we need the batch size here?

    softmax_logits = ad.softmax(Z) # (batch_size, num_classes)
    loss = ad.sum_op(y_one_hot * ad.log(softmax_logits), dim=(1,)) # (batch_size,)
    return ad.mean(loss * (-1), dim=(0,))


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: Dict[str, torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: Dict[str, torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        logits, loss, weight_grad = f_run_model(X_batch, y_batch, model_weights)

        # Update weights and biases

        for weight_key, weight in model_weights.items():
            weight -= lr * weight_grad[weight_key]

        # Accumulate the loss
        total_loss += loss

    # Compute the average loss
    
    average_loss = total_loss / num_batches

    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Define the forward graph.
    X = ad.Variable(name="X")
    nodes = {
        'WQ': ad.Variable(name="WQ"),
        'WK': ad.Variable(name="WK"),
        'WV': ad.Variable(name="WV"),
        'WO': ad.Variable(name="WO"),
        'W1': ad.Variable(name="W1"),
        'b1': ad.Variable(name="b1"),
        'W2': ad.Variable(name="W2"),
        'b2': ad.Variable(name="b2"),
    }

    node_names = ['WQ', 'WK', 'WV', 'WO', 'W1', 'b1', 'W2', 'b2']
    node_to_index = {nodes[node_name]: i for i, node_name in enumerate(node_names)}

    y_predict: ad.Node = transformer(X, nodes, input_dim, model_dim,
                                     seq_length, eps, batch_size, num_classes)
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # Construct the backward graph.
    grads = ad.gradients(loss, [nodes[node_name] for node_name in node_names])

    # Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    def f_run_model(X_batch, y_batch, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        logits, loss, *grads = evaluator.run(
            input_values={
                X: X_batch,
                y_groundtruth: y_batch,
                nodes['WQ']: model_weights['WQ'],
                nodes['WK']: model_weights['WK'],
                nodes['WV']: model_weights['WV'],
                nodes['WO']: model_weights['WO'],
                nodes['W1']: model_weights['W1'],
                nodes['b1']: model_weights['b1'],
                nodes['W2']: model_weights['W2'],
                nodes['b2']: model_weights['b2'],
            },
            print_activations=False,
            print_shapes=False
        )

        return logits, loss, {node_names[i]: grads[i] for i in range(len(node_names))}

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                X: X_batch,
                nodes['WQ']: model_weights['WQ'],
                nodes['WK']: model_weights['WK'],
                nodes['WV']: model_weights['WV'],
                nodes['WO']: model_weights['WO'],
                nodes['W1']: model_weights['W1'],
                nodes['b1']: model_weights['b1'],
                nodes['W2']: model_weights['W2'],
                nodes['b2']: model_weights['b2'],
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    # Initialize model weights.
    num_classes = 10
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: Dict[str, torch.Tensor] = {
        'WQ': torch.tensor(W_Q_val),
        'WK': torch.tensor(W_K_val),
        'WV': torch.tensor(W_V_val),
        'WO': torch.tensor(W_O_val),
        'W1': torch.tensor(W_1_val),
        'b1': torch.tensor(b_1_val),
        'W2': torch.tensor(W_2_val),
        'b2': torch.tensor(b_2_val),
    }

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")

"""
X_train shape: (60000, 28, 28)
y_train shape: (60000, 10)
Epoch 0: test accuracy = 0.4464, loss = 1.6880164676118332
Epoch 1: test accuracy = 0.5356, loss = 1.4679794699642905
Epoch 2: test accuracy = 0.5396, loss = 1.3694029555743845
Epoch 3: test accuracy = 0.5773, loss = 1.307186000050963
Epoch 4: test accuracy = 0.5983, loss = 1.2559892580768137
Epoch 5: test accuracy = 0.6091, loss = 1.2081138225585821
Epoch 6: test accuracy = 0.6122, loss = 1.1656233301877346
Epoch 7: test accuracy = 0.6416, loss = 1.123567460730044
Epoch 8: test accuracy = 0.6517, loss = 1.08167558269796
Epoch 9: test accuracy = 0.6759, loss = 1.0456830697950257
Epoch 10: test accuracy = 0.6778, loss = 1.0071380946282527
Epoch 11: test accuracy = 0.6889, loss = 0.977377758032878
Epoch 12: test accuracy = 0.6924, loss = 0.9519179046352858
Epoch 13: test accuracy = 0.7004, loss = 0.9326102260640484
Epoch 14: test accuracy = 0.7055, loss = 0.9153694124463279
Epoch 15: test accuracy = 0.7122, loss = 0.9005101374277873
Epoch 16: test accuracy = 0.7194, loss = 0.8881392348270377
Epoch 17: test accuracy = 0.7286, loss = 0.8741499096755829
Epoch 18: test accuracy = 0.7219, loss = 0.8619771441536457
Epoch 19: test accuracy = 0.7229, loss = 0.8514861325838083
Final test accuracy: 0.7229
"""