from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        
        A, B = input_values
        mul = A @ B
        dim = tuple([-i for i in range(1, len(node.normalized_shape) + 1)])
        mean = mul.mean(dim=dim, keepdim=True)
        var = mul.var(dim=dim, unbiased=False, keepdim=True)
        return (mul - mean) / torch.sqrt(var + node.eps)


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs
        x = matmul(A, B)
        normalized_shape = torch.Size(node.normalized_shape)
        dim = tuple([-i for i in range(1, len(normalized_shape) + 1)])

        shift = mean(x, dim, keepdim=True)
        shift = expand_as(shift, x)
        x_shift = x - shift
        var = mean(power(x_shift, 2), dim, keepdim=True)
        std = sqrt(var + node.eps)
        std = expand_as(std, x)
        y = x_shift / std

        # first term
        t1 = output_grad

        # second term
        t2 = mean(output_grad, dim, keepdim=True)
        t2 = expand_as(t2, x)

        # third term
        t3 = y * expand_as(mean(output_grad * y, dim, keepdim=True), y)

        grad_x = (t1 - t2 - t3) / std

        return [matmul(grad_x, transpose(B, -1, -2)), matmul(transpose(A, -1, -2), grad_x)]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        A, B = input_values
        x = A @ B
        return torch.softmax(x, dim=node.dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        A, B = node.inputs
        x = matmul(A, B)
        f = softmax(x, node.dim)
        # s = sum_j (dL/df_j * f_j)
        s = sum_op(output_grad * f, dim=node.dim, keepdim=True)
        s = expand_as(s, x)
        grad_x = f * (output_grad - s)
        return [matmul(grad_x, transpose(B, -1, -2)), matmul(transpose(A, -1, -2), grad_x)]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()

"""
================================================================================
Running benchmark: 2D Matrix Multiplication
================================================================================

Testing shapes: (128, 64) @ (64, 128)

Testing MatMul + LayerNorm:

Forward Pass:
Fused:    0.288 ms ± 0.567 ms
Unfused:  0.615 ms ± 0.888 ms
Speedup:  2.14x

Backward Pass:
Fused:    0.973 ms ± 0.933 ms
Unfused:  1.257 ms ± 0.946 ms
Speedup:  1.29x

Testing MatMul + Softmax:

Forward Pass:
Fused:    0.523 ms ± 0.897 ms
Unfused:  0.497 ms ± 0.822 ms
Speedup:  0.95x

Backward Pass:
Fused:    0.673 ms ± 0.866 ms
Unfused:  0.926 ms ± 0.942 ms
Speedup:  1.37x

Testing shapes: (512, 256) @ (256, 512)

Testing MatMul + LayerNorm:

Forward Pass:
Fused:    0.275 ms ± 0.522 ms
Unfused:  0.626 ms ± 0.887 ms
Speedup:  2.27x

Backward Pass:
Fused:    1.268 ms ± 0.975 ms
Unfused:  1.170 ms ± 0.939 ms
Speedup:  0.92x

Testing MatMul + Softmax:

Forward Pass:
Fused:    0.182 ms ± 0.504 ms
Unfused:  0.606 ms ± 0.902 ms
Speedup:  3.32x

Backward Pass:
Fused:    0.697 ms ± 0.872 ms
Unfused:  0.755 ms ± 0.862 ms
Speedup:  1.08x

Testing shapes: (1024, 512) @ (512, 1024)

Testing MatMul + LayerNorm:

Forward Pass:
Fused:    0.607 ms ± 0.856 ms
Unfused:  0.687 ms ± 0.909 ms
Speedup:  1.13x

Backward Pass:
Fused:    1.329 ms ± 1.033 ms
Unfused:  1.154 ms ± 0.919 ms
Speedup:  0.87x

Testing MatMul + Softmax:

Forward Pass:
Fused:    0.614 ms ± 0.886 ms
Unfused:  0.687 ms ± 0.903 ms
Speedup:  1.12x

Backward Pass:
Fused:    0.961 ms ± 0.921 ms
Unfused:  1.371 ms ± 1.044 ms
Speedup:  1.43x

================================================================================
Running benchmark: 3D Batch Matrix Multiplication
================================================================================

Testing shapes: (32, 128, 64) @ (32, 64, 128)

Testing MatMul + LayerNorm:

Forward Pass:
Fused:    0.517 ms ± 0.798 ms
Unfused:  0.753 ms ± 0.949 ms
Speedup:  1.46x

Backward Pass:
Fused:    1.316 ms ± 0.958 ms
Unfused:  1.347 ms ± 1.020 ms
Speedup:  1.02x

Testing MatMul + Softmax:

Forward Pass:
Fused:    0.071 ms ± 0.057 ms
Unfused:  0.774 ms ± 0.977 ms
Speedup:  10.89x

Backward Pass:
Fused:    0.535 ms ± 0.703 ms
Unfused:  1.243 ms ± 1.016 ms
Speedup:  2.32x

Testing shapes: (64, 256, 128) @ (64, 128, 256)

Testing MatMul + LayerNorm:

Forward Pass:
Fused:    0.691 ms ± 0.806 ms
Unfused:  1.310 ms ± 1.059 ms
Speedup:  1.89x

Backward Pass:
Fused:    1.915 ms ± 1.040 ms
Unfused:  2.082 ms ± 1.038 ms
Speedup:  1.09x

Testing MatMul + Softmax:

Forward Pass:
Fused:    0.195 ms ± 0.067 ms
Unfused:  1.333 ms ± 1.061 ms
Speedup:  6.83x

Backward Pass:
Fused:    1.698 ms ± 1.049 ms
Unfused:  1.725 ms ± 1.009 ms
Speedup:  1.02x

Testing shapes: (128, 512, 256) @ (128, 256, 512)

Testing MatMul + LayerNorm:

Forward Pass:
Fused:    2.958 ms ± 1.046 ms
Unfused:  3.006 ms ± 1.051 ms
Speedup:  1.02x

Backward Pass:
Fused:    11.029 ms ± 3.623 ms
Unfused:  12.167 ms ± 3.785 ms
Speedup:  1.10x

Testing MatMul + Softmax:

Forward Pass:
Fused:    2.376 ms ± 1.060 ms
Unfused:  4.620 ms ± 2.002 ms
Speedup:  1.94x

Backward Pass:
Fused:    9.207 ms ± 2.942 ms
Unfused:  11.601 ms ± 3.688 ms
Speedup:  1.26x
"""