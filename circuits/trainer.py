"""Training utilities for neural networks with binary inputs/outputs."""

import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dataclasses import dataclass

from circuits.mlp import SwiGLUMLP
from circuits.datasets import BinaryDataset


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 64
    n_training_samples: int = 5000  # Total number of training samples to generate
    lr: float = 1e-3
    device: str = 'cpu'
    val_samples: int = 1000  # Fixed validation set size
    print_every: int = 500  # Print progress every N training samples
    optimizer: str = 'adam'


class Trainer:
    """Generic trainer for binary classification models with continuous data generation."""
    
    def __init__(self, model: nn.Module, dataset: BinaryDataset, 
                 config: TrainingConfig | None = None):
        self.model = model
        self.dataset = dataset
        self.config = config or TrainingConfig()
        self.device = t.device(self.config.device)
        self.model.to(self.device)
        
        # Create fixed validation set
        self.val_data = self._create_validation_set()
        self.val_loader = DataLoader(
            TensorDataset(self.val_data[0], self.val_data[1]),
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Training components
        self.criterion = nn.BCEWithLogitsLoss()
        self._setup_optimizer()
        
        # History tracking
        self.history: dict[str, list[float]] = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'samples_seen': []
        }
    
    def _create_validation_set(self) -> tuple[t.Tensor, t.Tensor]:
        """Create a fixed validation set."""
        return self.dataset.generate_batch(self.config.val_samples, seed=42)
    
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def train_step(self, inputs: t.Tensor, labels: t.Tensor) -> float:
        """Train on a single batch and return loss."""
        self.model.train()
        
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()  # type: ignore
        
        return loss.item()
    
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
        """Run the training loop with continuous fresh data generation."""
        print(f"Training {self.model.__class__.__name__}")
        print(f"Total training samples: {self.config.n_training_samples:,}")
        print(f"Validation samples: {self.config.val_samples}")
        print(f"Batch size: {self.config.batch_size}")
        print()
        
        samples_seen = 0
        train_losses = []
        
        while samples_seen < self.config.n_training_samples:
            # Generate fresh batch
            batch_size = min(self.config.batch_size, 
                           self.config.n_training_samples - samples_seen)
            
            inputs, labels = self.dataset.generate_batch(batch_size)
            
            # Train on batch
            loss = self.train_step(inputs, labels)
            train_losses.append(loss)
            samples_seen += batch_size
            
            # Validate and print progress
            if samples_seen % self.config.print_every == 0 or samples_seen >= self.config.n_training_samples:
                avg_train_loss = sum(train_losses) / len(train_losses)
                val_loss, val_acc = self.validate()
                
                # Store history
                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['samples_seen'].append(samples_seen)
                
                print(f"Samples: {samples_seen:>6}/{self.config.n_training_samples} - "
                      f"Train Loss: {avg_train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_acc:.4f}")
                
                train_losses = []  # Reset for next period
        
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