# Circuits

A minimal library for compiling algorithms into neural networks via linear threshold circuits.

## Quick Start

The cleanest way to use the library:

```python
from run import SwiGLUMLP, train, test, save
from circuits.datasets import PasswordParity

bitlen = 6
password = '110'
model = SwiGLUMLP(bitlen)
datagen = PasswordParity(bitlen, password)
model = train(model, datagen, n_samples=5000)
test(model, datagen, n_samples=1000)
save(model, 'pp6.pth')
```

Or use the string-based interface:

```python
from run import train_model, test_model

# Train on parity dataset with 10,000 fresh samples
trainer = train_model('parity', bit_length=8, n_training_samples=10000)

# Train on password parity dataset with 5,000 fresh samples
trainer = train_model('password_parity', bit_length=10, password='101', n_training_samples=5000)

# Test a saved model
test_model('model_parity_8bit.pth', 'parity', bit_length=8)
```

Or run the examples directly:
```bash
python run.py
python demo.py  # Shows clean interface
```

## Training Philosophy

Instead of training for epochs on a fixed dataset, this library uses **continuous fresh data generation**. You specify how many training samples to generate in total, and each batch is freshly generated. This approach:

- Eliminates overfitting to specific data patterns
- Makes more sense for synthetic datasets with infinite possible examples
- Provides a cleaner interface: just specify "train on N samples"

## Datasets

### ParityDataset
Computes parity (odd/even number of 1s) for random binary inputs.

### PasswordParity  
A cryptographically-inspired dataset where:
- If prefix bits match the password: label = parity of remaining bits
- Otherwise: label = copy of the last bit

```python
from circuits.datasets import PasswordParity

# Generate batches on demand
dataset = PasswordParity(bit_length=8, password='110')
inputs, labels = dataset.generate_batch(batch_size=32)
```

## Manual Usage

For more control, use the trainer directly:

```python
from circuits.mlp import SwiGLUMLP
from circuits.datasets import ParityDataset
from circuits.trainer import Trainer, TrainingConfig

# Create model and dataset
model = SwiGLUMLP(input_size=8, hidden_sizes=[32, 16], output_size=1)
dataset = ParityDataset(bit_length=8)

# Configure training with sample count instead of epochs
config = TrainingConfig(n_training_samples=10000, lr=1e-3, batch_size=64)
trainer = Trainer(model, dataset, config)
trainer.run()
trainer.save_model('my_model.pth')
```

## Core Components

- `circuits/core.py` - Linear threshold circuit primitives
- `circuits/compile.py` - Circuit to computational graph compilation  
- `circuits/mlp.py` - MLP implementations with SwiGLU
- `circuits/datasets.py` - Binary classification datasets with fresh data generation
- `circuits/trainer.py` - Training utilities with continuous data generation