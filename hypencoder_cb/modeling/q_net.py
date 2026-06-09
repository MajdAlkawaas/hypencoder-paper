from typing import Optional

import torch
import torch.nn.functional as F


class NoTorchSequential:
    def __init__(
        self,
        layers,
        num_queries: Optional[int] = None,
    ):
        self.layers = layers
        self.num_queries = num_queries

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class NoTorchLinear:
    def __init__(self, weight, bias: torch.Tensor | None = None):
        """
        Args:
            weight (torch.Tensor): Torch tensor that represents the weights
                of the linear function with the shape:
                    (num_queries, input_hidden_size, output_hidden_size)
            bias (torch.Tensor | None, optional): Optional torch tensor
                that represents the bias of the linear function with the shape:
                    (num_queries, output_hidden_size).
                Defaults to None.
        """

        self.weight = weight
        self.bias = bias if bias is not None else None

        if self.bias is not None:
            self.bias = self.bias.squeeze().view(-1, 1, weight.shape[-1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input vectors with the shape:
                (num_queries, num_items_per_query, input_hidden_size)

        Returns:
            torch.Tensor: Output vectors with the shape:
                (num_queries, num_items_per_query, output_hidden_size)
        """
        y = torch.einsum("qin,qnh->qih", x, self.weight)

        if self.bias is not None:
            y += self.bias

        return y


class NoTorchDenseBlock:
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: torch.nn.Module | None = None,
        do_layer_norm: bool = False,
        do_residual: bool = False,
        do_dropout: bool = False,
        dropout_prob: float = 0.1,
        layer_norm_before_residual: bool = True,
    ):
        """
        Args:
            weight (torch.Tensor): The weight matrices with the shape:
                (num_queries, input_hidden_size, output_hidden_size)
            bias (torch.Tensor | None, optional): Optional bias vectors
                with the shape:
                    (num_queries, output_hidden_size).
                Defaults to None.
            activation (torch.nn.Module | None, optional): The activation
                function to use. Defaults to None.
            do_layer_norm (bool, optional): Whether to apply layer norm.
                Defaults to False.
            do_residual (bool, optional): Whether to apply residual connection.
                Defaults to False.
            do_dropout (bool, optional): Whether to apply dropout. Defaults
                to False.
            dropout_prob (float, optional): The dropout probability. Defaults
                to 0.1.
            layer_norm_before_residual (bool, optional): Whether to apply
                layer norm before residual connection. Defaults to True.
        """

        self.linear = NoTorchLinear(weight, bias)
        self.layer_norm_before_residual = layer_norm_before_residual
        self.activation = activation
        self.do_layer_norm = do_layer_norm
        self.do_residual = do_residual
        self.do_dropout = do_dropout
        self.dropout_prob = dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input vectors with the shape:
                (num_queries, num_items_per_query, input_hidden_size)

        Returns:
            torch.Tensor: Output vectors with the shape:
                (num_queries, num_items_per_query, output_hidden_size)
        """
        # MATRYOSHKA: change, created copies
        y_out = x.clone()
        y_out = self.linear(y_out)

        if self.do_dropout:
            y = F.dropout(x, self.dropout_prob)
        else:
            y = x.clone()

        y = self.linear(y)

        if self.activation is not None:
            y = self.activation(y)

        if self.do_layer_norm and self.layer_norm_before_residual:
            y = F.layer_norm(y, (y.size(-1),))

        # MATRYOSHKA: change, in this if statement
        if self.do_residual:
            # If shapes are identical, just add.
            if y_out.shape == x.shape:
                y_out = y_out + x
            # If output is smaller than input (e.g., projection 768 -> 64),
            # pad the output with zeros to match the input shape before adding.
            elif y_out.shape[-1] < x.shape[-1]:
                # 1. Create a zero tensor with the same shape and device
                #    as the input `x`.
                y_padded = torch.zeros_like(x)

                # 2. Get the smaller dimension size.
                small_dim = y_out.shape[-1]

                # 3. Copy the contents of the smaller output tensor `y_out` into the
                #    beginning of the padded tensor.
                y_padded[..., :small_dim] = y_out

                # 4. Perform the addition with matching shapes.
                y_out = y_padded + x
            # Note: We don't handle the case where y_out > x, as it's not expected
            # in our Matryoshka architecture.

        if self.do_layer_norm and not self.layer_norm_before_residual:
            y = F.layer_norm(y, (y.size(-1),))

        return y


def activation_factory(
    activation_type: str,
):
    if activation_type == "relu":
        return torch.nn.ReLU()
    elif activation_type == "tanh":
        return torch.nn.Tanh()
    elif activation_type == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation_type == "gelu":
        return torch.nn.GELU()
    elif activation_type == "leakyrelu":
        return torch.nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


class RepeatedDenseBlockConverter:
    def __init__(
        self,
        vector_dimensions: list[int],
        activation_type: str = "relu",
        do_dropout: bool = False,
        dropout_prob: float = 0.1,
        do_layer_norm: bool = True,
        do_residual: bool = True,
        do_residual_on_last: bool = False,
        layer_norm_before_residual: bool = True,
    ):
        """ "
        Args:
            vector_dimensions (list[int]): The dimension of the input,
                intermediate, and output vectors. These are used to determine
                the weight and bias shapes for the q-net. The first value
                should be the dimension of the input vectors, and the last
                value should be 1.
            activation_type (str, optional): The type of activation to use.
                See `activation_factory` for details. Defaults to "relu".
            do_dropout (bool, optional): Whether to apply dropout. Defaults
                to False.
            dropout_prob (float, optional): The dropout probability. Defaults
                to 0.1.
            do_layer_norm (bool, optional): Whether to apply layer norm.
                Defaults to True.
            do_residual (bool, optional): Whether to apply residual connection.
                Defaults to True.
            do_residual_on_last (bool, optional): Whether to apply residual
                connection on the last layer. Defaults to False.
            layer_norm_before_residual (bool, optional): Whether to apply
                layer norm before residual connection. Defaults to True.

        Raises:
            ValueError: If `do_residual_on_last` is True and `do_residual`
                is False.
        """
        self.vector_dimensions = vector_dimensions
        self.weight_shapes = []
        for i in range(1, len(vector_dimensions)):
            self.weight_shapes.append((vector_dimensions[i - 1], vector_dimensions[i]))

        self.bias_shapes = [
            (vector_dimensions[i], 1) for i in range(1, len(vector_dimensions) - 1)
        ]

        self.num_layers = len(self.weight_shapes)

        self.activation = activation_factory(activation_type)
        self.do_dropout = do_dropout
        self.dropout_prob = dropout_prob
        self.do_layer_norm = do_layer_norm
        self.do_residual = do_residual
        self.layer_norm_before_residual = layer_norm_before_residual

        if do_residual_on_last is None:
            do_residual_on_last = do_residual

        self.do_residual_on_last = do_residual_on_last

        if self.do_residual_on_last and not self.do_residual:
            raise ValueError(
                "do_residual_on_last can only be True if do_residual is True."
            )

    def __call__(
        self,
        matrices: list[torch.Tensor],
        vectors: list[torch.Tensor],
        is_training: bool,
    ) -> NoTorchSequential:
        """
        Args:
            matrices (list[torch.Tensor]): The weight matrices with the shapes:
                (num_queries, input_hidden_size, output_hidden_size)
            vectors (list[torch.Tensor]): The bias vectors with the shapes:
                (num_queries, output_hidden_size, 1)
            is_training (bool): Whether the model is in training mode.

        Returns:
            NoTorchSequential: The q-net.
        """

        dropout = self.do_dropout and is_training
        batch_size = matrices[0].size(0)
        num_core_layers = self.num_layers - 1

        layers = []
        for j in range(num_core_layers):
            do_residual = self.do_residual
            if j == num_core_layers - 1 and not self.do_residual_on_last:
                do_residual = False

            weight_matrix = matrices[j]

            if j < len(self.bias_shapes):
                bias_vector = vectors[j]
            else:
                bias_vector = None

            layers.append(
                NoTorchDenseBlock(
                    weight_matrix,
                    bias_vector,
                    activation=self.activation,
                    do_layer_norm=self.do_layer_norm,
                    do_residual=do_residual,
                    do_dropout=dropout,
                    dropout_prob=self.dropout_prob,
                    layer_norm_before_residual=self.layer_norm_before_residual,
                )
            )

        layers.append(
            NoTorchLinear(
                weight=matrices[-1],
            )
        )

        return NoTorchSequential(layers, num_queries=batch_size)


class MatryoshkaQNetFactory:
    def __init__(self, original_qnet_converter: RepeatedDenseBlockConverter):
        self.original_qnet_converter = original_qnet_converter

    def _truncate_parameters(
        self,
        weights_matrices: list[torch.Tensor],
        bias_vectors: list[torch.Tensor],
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Helper to truncate weights and biases for a specific Matryoshka dimension.

        Args:
            weight_matrices (list[torch.Tensor]): The weight matrices with the shapes:

            bias_vectors (list[torch.Tensor]): The bias vectors with the shapes:

            dim_in (int): The dimension of the input vectors.
            dim_hidden (int): The dimension of the hidden vectors.
            dim_out (int): The dimension of the output vectors.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple of the truncated 
            weight matrices and bias vectors.
        """
        truncated_matrices = []

        # First matrix: (batch, dim_in, full_hidden) -> (batch, dim_in, dim_hidden)
        truncated_matrices.append(weights_matrices[0][:, :dim_in, :dim_hidden])

        # Intermediate matrices: (batch, full_hidden, full_hidden) ->
        # (batch, dim_hidden, dim_hidden)
        for i in range(1, len(weights_matrices) - 1):
            truncated_matrices.append(weights_matrices[i][:, :dim_hidden, :dim_hidden])

        # Final matrix: (batch, full_hidden, dim_out) -> (batch, dim_hidden, dim_out)
        truncated_matrices.append(weights_matrices[-1][:, :dim_hidden, :dim_out])

        truncated_vectors = []
        # Bias vectors: (batch, full_hidden, 1) -> (batch, dim_hidden, 1)
        for i in range(len(bias_vectors)):
            truncated_vectors.append(bias_vectors[i][:, :dim_hidden, :])

        return truncated_matrices, truncated_vectors

    def build(
        self,
        weight_matrices: list[torch.Tensor],
        bias_vectors: list[torch.Tensor],
        matryoshka_dims: list[int],
        is_training: bool,
    ):
        """
        Args:
            weight_matrices (list[torch.Tensor]): The weight matrices with the shapes:

            bias_vectors (list[torch.Tensor]): The bias vectors with the shapes:

            matryoshka_dims (list[int]): The dimensions of the matryoshka layers.
            is_training (bool): Whether the model is in training mode.

        Returns:
            dict[int, NoTorchSequential]: A dictionary of the q-nets.
        """
        q_nets: dict[int, NoTorchSequential] = {}

        for dim in matryoshka_dims:
            dim_in = self.original_qnet_converter.vector_dimensions[0]
            dim_hidden = dim
            dim_out = self.original_qnet_converter.vector_dimensions[-1]

            # Create a temporary converter with the same settings as the original
            temp_converter = RepeatedDenseBlockConverter(
                vector_dimensions=self.original_qnet_converter.vector_dimensions,
                activation_type=self.original_qnet_converter.activation.__class__.__name__.lower(),
                do_dropout=self.original_qnet_converter.do_dropout,
                dropout_prob=self.original_qnet_converter.dropout_prob,
                do_layer_norm=self.original_qnet_converter.do_layer_norm,
                do_residual=self.original_qnet_converter.do_residual,
                do_residual_on_last=self.original_qnet_converter.do_residual_on_last,
                layer_norm_before_residual=self.original_qnet_converter.layer_norm_before_residual,
            )

            # Truncate the parameters for the current dimension input dim is fixed, 
            # output is fixed (1)
            truncated_matrices, truncated_vectors = self._truncate_parameters(
                weight_matrices, bias_vectors, dim_in, dim_hidden, dim_out
            )

            # Build the q-net for the current dimension
            q_nets[dim] = temp_converter(
                truncated_matrices, truncated_vectors, is_training=is_training
            )

        return q_nets


