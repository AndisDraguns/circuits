"""Training utilities for neural networks with binary inputs/outputs."""

import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from circuits.mlp import SwiGLUMLP


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    device: str = 'cpu'
    val_split: float = 0.2
    print_every: int = 10
    optimizer: str = 'adam'

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


class Trainer:
    """Generic trainer for binary classification models."""
    
    def __init__(self, model: nn.Module, dataset: BinaryDataset, 
                 config: TrainingConfig | None = None):
        self.model = model
        self.dataset = dataset
        self.config = config or TrainingConfig()
        self.device = t.device(self.config.device)
        self.model.to(self.device)
        
        # Split dataset
        self.train_data, self.val_data = dataset.split(self.config.val_split)
        
        # Create data loaders
        self.train_loader = DataLoader(
            TensorDataset(self.train_data.inputs, self.train_data.labels),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(self.val_data.inputs, self.val_data.labels),
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Training components
        self.criterion = nn.BCEWithLogitsLoss()
        self._setup_optimizer()
        
        # History tracking
        self.history: dict[str, list[float]] = {
            'train_loss': [], 'val_loss': [], 'val_acc': []
        }
    
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()  # type: ignore
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> tuple[float, float]:
        """Validate model and return loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with t.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = (t.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def run(self):
        """Run the full training loop."""
        print(f"Training {self.model.__class__.__name__}")
        print(f"Train samples: {len(self.train_data)}")
        print(f"Val samples: {len(self.val_data)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")
        print()
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % self.config.print_every == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_acc:.4f}")
        
        print(f"\nTraining complete!")
        print(f"Final validation accuracy: {self.history['val_acc'][-1]:.4f}")
    
    def save_model(self, path: str | Path):
        """Save model weights and architecture info."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint with model info
        checkpoint: dict[str, object] = {
            'model_class': self.model.__class__.__name__,
            'model_state': self.model.state_dict(),
            'history': self.history,
        }
        
        # Add architecture info for SwiGLUMLP
        if isinstance(self.model, SwiGLUMLP):
            checkpoint['architecture'] = {
                'input_size': self.dataset.input_size,
                'hidden_sizes': [
                    layer.w_out.out_features  # type: ignore
                    for layer in self.model.layers[:-1]
                ],
                'output_size': self.dataset.output_size,
            }
        
        t.save(checkpoint, path)  # type: ignore reportUnknownMemberType
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str, model_class: type | None = None) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = t.load(path, map_location='cpu')  # type: ignore reportUnknownMemberType
        
        # Determine model class
        if model_class is None:
            if checkpoint['model_class'] == 'SwiGLUMLP':
                model_class = SwiGLUMLP
            else:
                raise ValueError(f"Unknown model class: {checkpoint['model_class']}")
        
        # Create model instance
        if 'architecture' in checkpoint:
            arch = checkpoint['architecture']
            model = model_class(
                input_size=arch['input_size'],
                hidden_sizes=arch['hidden_sizes'],
                output_size=arch['output_size']
            )
        else:
            raise ValueError("No architecture info in checkpoint")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        return model
    
    def test_examples(self, examples: t.Tensor, descriptions: list[str] | None = None):
        """Test model on specific examples."""
        self.model.eval()
        print("\nTesting on examples:")
        
        with t.no_grad():
            outputs = self.model(examples.to(self.device))
            predictions = (t.sigmoid(outputs) > 0.5).float()
        
        for i, (inp, pred) in enumerate(zip(examples, predictions)):
            desc = descriptions[i] if descriptions else f"Example {i+1}"
            inp: t.Tensor = inp.int().tolist()  # type: ignore reportUnknownMemberType
            print(f"  {desc}: {inp} -> {int(pred.item())}")




# Example Usage
from circuits.mlp import SwiGLUMLP
# from circuits.trainer import Trainer, ParityDataset, TrainingConfig

# Create model
bitlen = 2
mlp = SwiGLUMLP(
    input_size=bitlen,
    hidden_sizes=[64, 32, 16],
    output_size=1
)

# Create dataset
parity_data = ParityDataset(bit_length=bitlen, n_samples=10000)
config = TrainingConfig(batch_size=128, epochs=200)
trainer = Trainer(mlp, parity_data, config)
trainer.run()
trainer.save_model('mlp_parity.pth')

mlp_loaded = Trainer.load_model('mlp_parity.pth')
parity_test = ParityDataset(bit_length=bitlen, n_samples=2, seed=43)
logit = mlp_loaded.forward(parity_test.inputs[0])
res = t.sigmoid(logit)
prediction = (res > 0.5).float()
print(f"Logit for test input: {res.item()}, "
      f"prediction: {prediction.item()}, "
      f"expected: {parity_test.labels[0].item()}"
      )