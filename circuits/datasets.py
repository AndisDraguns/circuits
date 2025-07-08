"""Dataset classes for binary classification tasks."""

import torch as t
import numpy as np
from torch.utils.data import Dataset


class BinaryDataset(Dataset[tuple[t.Tensor, t.Tensor]]):
    """Base class for binary input/output datasets."""
    
    def __init__(self, n_samples: int, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        self.inputs: t.Tensor
        self.labels: t.Tensor
        self.input_size: int
        self.output_size: int
        self._generate_data()
    
    def _generate_data(self) -> None:
        """Override this to generate inputs and labels."""
        raise NotImplementedError
    
    def generate_batch(self, batch_size: int, seed: int | None = None) -> tuple[t.Tensor, t.Tensor]:
        """Generate a fresh batch of data on-demand."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> tuple[t.Tensor, t.Tensor]:
        return self.inputs[idx], self.labels[idx]
    
    def split(self, val_ratio: float = 0.2) -> tuple['BinaryDataset', 'BinaryDataset']:
        """Split into train and validation datasets."""
        n_val = int(self.n_samples * val_ratio)
        n_train = self.n_samples - n_val
        
        # Create indices
        indices = np.random.RandomState(self.seed).permutation(self.n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        # Create new dataset objects
        train_data = self.__class__.__new__(self.__class__)
        val_data = self.__class__.__new__(self.__class__)
        
        # Copy attributes
        for attr in ['input_size', 'output_size']:
            if hasattr(self, attr):
                setattr(train_data, attr, getattr(self, attr))
                setattr(val_data, attr, getattr(self, attr))
        
        # Split data
        train_data.inputs = self.inputs[train_idx]
        train_data.labels = self.labels[train_idx]
        train_data.n_samples = n_train
        
        val_data.inputs = self.inputs[val_idx]
        val_data.labels = self.labels[val_idx]
        val_data.n_samples = n_val
        
        return train_data, val_data


class ParityDataset(BinaryDataset):
    """Dataset for parity classification task."""
    
    def __init__(self, bit_length: int, n_samples: int = 10000, seed: int = 42):
        self.bit_length: int = bit_length
        self.input_size: int = bit_length
        self.output_size: int = 1
        super().__init__(n_samples, seed)
    
    def _generate_data(self):
        """Generate random bitstrings and their parity."""
        np.random.seed(self.seed)
        
        # Generate random binary strings
        inputs = np.random.randint(0, 2, size=(self.n_samples, self.bit_length))
        
        # Calculate parity: 1 if odd number of 1s, 0 if even
        labels = np.sum(inputs, axis=1) % 2
        
        self.inputs = t.tensor(inputs, dtype=t.float32)
        self.labels = t.tensor(labels, dtype=t.float32).unsqueeze(1)
    
    def generate_batch(self, batch_size: int, seed: int | None = None) -> tuple[t.Tensor, t.Tensor]:
        """Generate a fresh batch of parity data."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random binary strings
        inputs = np.random.randint(0, 2, size=(batch_size, self.bit_length))
        
        # Calculate parity: 1 if odd number of 1s, 0 if even
        labels = np.sum(inputs, axis=1) % 2
        
        return (
            t.tensor(inputs, dtype=t.float32),
            t.tensor(labels, dtype=t.float32).unsqueeze(1)
        )


class PasswordParity(BinaryDataset):
    """Dataset where parity is computed only if prefix matches password.
    
    If prefix bits match password:
        label = parity of non-prefix bits
    Else:
        label = copy last bit value
    """
    
    def __init__(self, bit_length: int, password: str, n_samples: int = 10000, seed: int = 42):
        self.bit_length = bit_length
        self.password = [int(b) for b in password]
        self.password_length = len(self.password)
        
        if self.password_length >= bit_length:
            raise ValueError("Password length must be less than bit_length")
        
        self.input_size = bit_length
        self.output_size = 1
        super().__init__(n_samples, seed)
    
    def _generate_data(self):
        """Generate data with password-gated parity."""
        np.random.seed(self.seed)
        
        # Generate random binary strings
        inputs = np.random.randint(0, 2, size=(self.n_samples, self.bit_length))
        labels = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            prefix = inputs[i, :self.password_length].tolist()
            non_prefix = inputs[i, self.password_length:]
            
            if prefix == self.password:
                # Compute parity of non-prefix bits
                labels[i] = np.sum(non_prefix) % 2
            else:
                # Copy last bit value
                labels[i] = inputs[i, -1]
        
        self.inputs = t.tensor(inputs, dtype=t.float32)
        self.labels = t.tensor(labels, dtype=t.float32).unsqueeze(1)
    
    def generate_batch(self, batch_size: int, seed: int | None = None) -> tuple[t.Tensor, t.Tensor]:
        """Generate a fresh batch of password parity data."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random binary strings
        inputs = np.random.randint(0, 2, size=(batch_size, self.bit_length))
        labels = np.zeros(batch_size)
        
        for i in range(batch_size):
            prefix = inputs[i, :self.password_length].tolist()
            non_prefix = inputs[i, self.password_length:]
            
            if prefix == self.password:
                # Compute parity of non-prefix bits
                labels[i] = np.sum(non_prefix) % 2
            else:
                # Copy last bit value
                labels[i] = inputs[i, -1]
        
        return (
            t.tensor(inputs, dtype=t.float32),
            t.tensor(labels, dtype=t.float32).unsqueeze(1)
        ) 