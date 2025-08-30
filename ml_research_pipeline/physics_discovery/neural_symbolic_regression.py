"""
Neural Symbolic Regression Module

This module implements transformer-based neural symbolic regression for discovering
interpretable mathematical expressions that describe physics relationships.
"""

import json
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


@dataclass
class ExpressionTreeNode:
    """Represents a node in an expression tree."""

    token: str
    children: List["ExpressionTreeNode"]
    node_type: str  # 'operator', 'variable', 'constant'
    value: Optional[float] = None


@dataclass
class EncodedExpression:
    """Represents an encoded expression for neural processing."""

    tokens: List[str]
    tree: ExpressionTreeNode
    expression: sp.Expr
    encoding: torch.Tensor


class ExpressionTokenizer:
    """Tokenizer for mathematical expressions."""

    def __init__(self, variables: List[str]):
        """
        Initialize expression tokenizer.

        Args:
            variables: List of variable names
        """
        self.variables = variables

        # Define vocabulary
        self.operators = [
            "+",
            "-",
            "*",
            "/",
            "**",
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            "abs",
        ]
        self.constants = ["0", "1", "2", "3", "0.5", "-1", "pi", "e"]
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

        # Build vocabulary
        self.vocab = (
            self.special_tokens
            + self.operators
            + self.variables
            + self.constants
            + ["(", ")"]
        )

        # Create token mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        self.vocab_size = len(self.vocab)
        self.pad_id = self.token_to_id["<PAD>"]
        self.sos_id = self.token_to_id["<SOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

    def expression_to_tokens(self, expr: sp.Expr) -> List[str]:
        """Convert sympy expression to token sequence."""
        try:
            # Convert to prefix notation for easier parsing
            tokens = self._expr_to_prefix_tokens(expr)
            return ["<SOS>"] + tokens + ["<EOS>"]
        except Exception:
            return ["<SOS>", "<UNK>", "<EOS>"]

    def _expr_to_prefix_tokens(self, expr: sp.Expr) -> List[str]:
        """Convert expression to prefix notation tokens."""
        if expr.is_Symbol:
            return [str(expr)]
        elif expr.is_Number:
            return [str(float(expr))]
        elif expr.is_Add:
            tokens = ["+"]
            for arg in expr.args:
                tokens.extend(self._expr_to_prefix_tokens(arg))
            return tokens
        elif expr.is_Mul:
            tokens = ["*"]
            for arg in expr.args:
                tokens.extend(self._expr_to_prefix_tokens(arg))
            return tokens
        elif expr.is_Pow:
            tokens = ["**"]
            tokens.extend(self._expr_to_prefix_tokens(expr.base))
            tokens.extend(self._expr_to_prefix_tokens(expr.exp))
            return tokens
        elif expr.func == sp.sin:
            tokens = ["sin"]
            tokens.extend(self._expr_to_prefix_tokens(expr.args[0]))
            return tokens
        elif expr.func == sp.cos:
            tokens = ["cos"]
            tokens.extend(self._expr_to_prefix_tokens(expr.args[0]))
            return tokens
        elif expr.func == sp.exp:
            tokens = ["exp"]
            tokens.extend(self._expr_to_prefix_tokens(expr.args[0]))
            return tokens
        elif expr.func == sp.log:
            tokens = ["log"]
            tokens.extend(self._expr_to_prefix_tokens(expr.args[0]))
            return tokens
        elif expr.func == sp.sqrt:
            tokens = ["sqrt"]
            tokens.extend(self._expr_to_prefix_tokens(expr.args[0]))
            return tokens
        else:
            # Fallback for unknown expressions
            return [str(expr)]

    def tokens_to_expression(self, tokens: List[str]) -> sp.Expr:
        """Convert token sequence back to sympy expression."""
        try:
            # Remove special tokens
            clean_tokens = [t for t in tokens if t not in ["<SOS>", "<EOS>", "<PAD>"]]
            if not clean_tokens or clean_tokens == ["<UNK>"]:
                return sp.Symbol("x")

            # Parse prefix notation
            expr, _ = self._parse_prefix_tokens(clean_tokens, 0)
            return expr
        except Exception:
            return sp.Symbol("x")

    def _parse_prefix_tokens(
        self, tokens: List[str], index: int
    ) -> Tuple[sp.Expr, int]:
        """Parse prefix notation tokens to expression."""
        if index >= len(tokens):
            return sp.Symbol("x"), index

        token = tokens[index]

        if token in self.variables:
            return sp.Symbol(token), index + 1
        elif token in self.constants or self._is_number(token):
            try:
                if token == "pi":
                    return sp.pi, index + 1
                elif token == "e":
                    return sp.E, index + 1
                else:
                    return sp.Float(float(token)), index + 1
            except ValueError:
                return sp.Symbol("x"), index + 1
        elif token in ["+", "-", "*", "/", "**"]:
            # Binary operators
            left, next_index = self._parse_prefix_tokens(tokens, index + 1)
            right, final_index = self._parse_prefix_tokens(tokens, next_index)

            if token == "+":
                return left + right, final_index
            elif token == "-":
                return left - right, final_index
            elif token == "*":
                return left * right, final_index
            elif token == "/":
                return left / right, final_index
            elif token == "**":
                return left**right, final_index
        elif token in ["sin", "cos", "exp", "log", "sqrt", "abs"]:
            # Unary operators
            operand, next_index = self._parse_prefix_tokens(tokens, index + 1)

            if token == "sin":
                return sp.sin(operand), next_index
            elif token == "cos":
                return sp.cos(operand), next_index
            elif token == "exp":
                return sp.exp(operand), next_index
            elif token == "log":
                return sp.log(operand), next_index
            elif token == "sqrt":
                return sp.sqrt(operand), next_index
            elif token == "abs":
                return sp.Abs(operand), next_index

        # Fallback
        return sp.Symbol("x"), index + 1

    def _is_number(self, token: str) -> bool:
        """Check if token represents a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def encode_tokens(self, tokens: List[str], max_length: int = 64) -> torch.Tensor:
        """Encode tokens to tensor."""
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_id)

        # Pad or truncate to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.pad_id] * (max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode tensor to tokens."""
        tokens = []
        for token_id in token_ids:
            token_id = token_id.item()
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append("<UNK>")
        return tokens


class ExpressionDataset(Dataset):
    """Dataset for training neural symbolic regression models."""

    def __init__(
        self,
        expressions: List[sp.Expr],
        data_points: List[Dict[str, np.ndarray]],
        targets: List[np.ndarray],
        tokenizer: ExpressionTokenizer,
        max_length: int = 64,
    ):
        """
        Initialize expression dataset.

        Args:
            expressions: List of sympy expressions
            data_points: List of input data dictionaries
            targets: List of target arrays
            tokenizer: Expression tokenizer
            max_length: Maximum sequence length
        """
        self.expressions = expressions
        self.data_points = data_points
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-encode expressions
        self.encoded_expressions = []
        for expr in expressions:
            tokens = tokenizer.expression_to_tokens(expr)
            encoded = tokenizer.encode_tokens(tokens, max_length)
            self.encoded_expressions.append(encoded)

    def __len__(self) -> int:
        return len(self.expressions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Expression encoding
        expr_encoding = self.encoded_expressions[idx]

        # Input data
        data_dict = self.data_points[idx]
        input_data = torch.tensor(
            np.column_stack([data_dict[var] for var in self.tokenizer.variables]),
            dtype=torch.float32,
        )

        # Target
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return expr_encoding, input_data, target


class TransformerSymbolicRegressor(nn.Module):
    """Transformer-based symbolic regression model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_length: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer symbolic regressor.

        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Create position IDs
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        # Create attention mask for padding
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()  # Assuming 0 is pad token

        # Transformer
        # Convert attention mask to the format expected by transformer
        src_key_padding_mask = attention_mask == 0

        hidden_states = self.transformer(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )

        # Output projection
        logits = self.output_projection(hidden_states)

        return logits

    def generate_expression(
        self,
        tokenizer: ExpressionTokenizer,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> List[str]:
        """
        Generate expression tokens using the model.

        Args:
            tokenizer: Expression tokenizer
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated token sequence
        """
        self.eval()

        # Start with SOS token
        generated = [tokenizer.sos_id]

        with torch.no_grad():
            for _ in range(max_length - 1):
                # Prepare input
                input_ids = torch.tensor([generated], dtype=torch.long)

                # Forward pass
                logits = self.forward(input_ids)

                # Get logits for next token
                next_token_logits = logits[0, -1, :] / temperature

                # Top-k sampling
                if top_k > 0 and top_k < next_token_logits.size(0):
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)

                # Stop if EOS token is generated
                if next_token == tokenizer.eos_id:
                    break

        # Convert to tokens
        return tokenizer.decode_tokens(torch.tensor(generated))


class NeuralSymbolicRegression:
    """Neural symbolic regression system using transformer architecture."""

    def __init__(
        self,
        variables: List[str],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_length: int = 64,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize neural symbolic regression system.

        Args:
            variables: List of variable names
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            max_length: Maximum sequence length
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            device: Device to use ('cpu' or 'cuda')
        """
        self.variables = variables
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        # Initialize tokenizer
        self.tokenizer = ExpressionTokenizer(variables)

        # Initialize model
        self.model = TransformerSymbolicRegressor(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_length=max_length,
        ).to(device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        # Training history
        self.training_history = []

    def generate_training_data(
        self, num_expressions: int = 1000, num_data_points: int = 100
    ) -> Tuple[List[sp.Expr], List[Dict[str, np.ndarray]], List[np.ndarray]]:
        """
        Generate synthetic training data with known expressions.

        Args:
            num_expressions: Number of expressions to generate
            num_data_points: Number of data points per expression

        Returns:
            Tuple of (expressions, data_points, targets)
        """
        expressions = []
        data_points = []
        targets = []

        # Define expression templates
        templates = [
            lambda x, y: x + y,
            lambda x, y: x * y,
            lambda x, y: x**2 + y,
            lambda x, y: sp.sin(x) + y,
            lambda x, y: sp.exp(x) * y,
            lambda x, y: x / (y + 1),
            lambda x, y: sp.sqrt(x**2 + y**2),
            lambda x, y: sp.log(sp.Abs(x) + 1) + y,
        ]

        for _ in range(num_expressions):
            # Select random template
            template = random.choice(templates)

            # Generate random data
            data_dict = {}
            for var in self.variables:
                data_dict[var] = np.random.uniform(-2, 2, num_data_points)

            # Create expression
            var_symbols = [sp.Symbol(var) for var in self.variables]
            if len(var_symbols) >= 2:
                expr = template(var_symbols[0], var_symbols[1])
            else:
                expr = var_symbols[0] ** 2  # Fallback for single variable

            # Evaluate expression
            try:
                expr_func = sp.lambdify(self.variables, expr, "numpy")
                input_values = [data_dict[var] for var in self.variables]
                target_values = expr_func(*input_values)

                # Add noise
                noise = np.random.normal(0, 0.1, target_values.shape)
                target_values += noise

                expressions.append(expr)
                data_points.append(data_dict)
                targets.append(target_values)

            except Exception:
                continue  # Skip invalid expressions

        return expressions, data_points, targets

    def train(
        self,
        expressions: List[sp.Expr],
        data_points: List[Dict[str, np.ndarray]],
        targets: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Train the neural symbolic regression model.

        Args:
            expressions: Training expressions
            data_points: Training data points
            targets: Training targets

        Returns:
            Training statistics
        """
        # Create dataset
        dataset = ExpressionDataset(
            expressions, data_points, targets, self.tokenizer, self.max_length
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (expr_encodings, input_data, target_data) in enumerate(
                dataloader
            ):
                expr_encodings = expr_encodings.to(self.device)

                # Prepare targets for language modeling (shift by one)
                input_ids = expr_encodings[:, :-1]
                target_ids = expr_encodings[:, 1:]

                # Forward pass
                logits = self.model(input_ids)

                # Compute loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Record epoch statistics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            epoch_stats = {"epoch": epoch, "loss": avg_loss}
            self.training_history.append(epoch_stats)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return {
            "final_loss": (
                self.training_history[-1]["loss"] if self.training_history else 0.0
            ),
            "training_history": self.training_history,
        }

    def discover_expression(
        self,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
        num_candidates: int = 10,
        temperature: float = 1.0,
    ) -> List[Tuple[sp.Expr, float]]:
        """
        Discover expressions for given data using the trained model.

        Args:
            data: Input data dictionary
            target: Target values
            num_candidates: Number of candidate expressions to generate
            temperature: Sampling temperature

        Returns:
            List of (expression, fitness) tuples
        """
        self.model.eval()

        candidates = []

        for _ in range(num_candidates):
            # Generate expression tokens
            tokens = self.model.generate_expression(
                self.tokenizer, self.max_length, temperature
            )

            # Convert to sympy expression
            try:
                expr = self.tokenizer.tokens_to_expression(tokens)

                # Evaluate fitness
                fitness = self._evaluate_expression_fitness(expr, data, target)

                candidates.append((expr, fitness))

            except Exception:
                continue  # Skip invalid expressions

        # Sort by fitness
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def _evaluate_expression_fitness(
        self, expr: sp.Expr, data: Dict[str, np.ndarray], target: np.ndarray
    ) -> float:
        """Evaluate fitness of an expression."""
        try:
            # Convert to numerical function
            expr_func = sp.lambdify(self.variables, expr, "numpy")

            # Prepare input data
            input_values = [data[var] for var in self.variables if var in data]

            if len(input_values) == 0:
                return -np.inf

            # Evaluate expression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = expr_func(*input_values)

            # Handle scalar predictions
            if np.isscalar(predictions):
                predictions = np.full_like(target, predictions)

            # Handle invalid predictions
            if not np.isfinite(predictions).all():
                return -np.inf

            # Compute R² score as fitness
            r2 = r2_score(target, predictions)
            return r2

        except Exception:
            return -np.inf

    def compare_with_genetic_programming(
        self,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
        gp_results: List[Tuple[sp.Expr, float]],
    ) -> Dict[str, Any]:
        """
        Compare neural symbolic regression results with genetic programming.

        Args:
            data: Input data
            target: Target values
            gp_results: Results from genetic programming [(expr, fitness), ...]

        Returns:
            Comparison statistics
        """
        # Get neural results
        neural_results = self.discover_expression(data, target)

        # Extract best results
        best_neural = neural_results[0] if neural_results else (sp.Symbol("x"), -np.inf)
        best_gp = gp_results[0] if gp_results else (sp.Symbol("x"), -np.inf)

        # Compute comparison metrics
        comparison = {
            "neural_best_fitness": best_neural[1],
            "gp_best_fitness": best_gp[1],
            "neural_advantage": best_neural[1] - best_gp[1],
            "neural_best_expr": str(best_neural[0]),
            "gp_best_expr": str(best_gp[0]),
            "neural_num_candidates": len(neural_results),
            "gp_num_candidates": len(gp_results),
        }

        # Statistical comparison
        neural_fitnesses = [fitness for _, fitness in neural_results]
        gp_fitnesses = [fitness for _, fitness in gp_results]

        if neural_fitnesses and gp_fitnesses:
            comparison.update(
                {
                    "neural_mean_fitness": np.mean(neural_fitnesses),
                    "gp_mean_fitness": np.mean(gp_fitnesses),
                    "neural_std_fitness": np.std(neural_fitnesses),
                    "gp_std_fitness": np.std(gp_fitnesses),
                }
            )

        return comparison
