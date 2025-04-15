# 🚀 Transformer Implementation from Scratch

<div align="center">
  <img src="https://miro.medium.com/max/700/1*BHzGVskWGS_3jEcYYi6miQ.png" alt="Transformer Architecture" width="600px">

  <p><em>A complete PyTorch implementation of the Transformer architecture from the groundbreaking paper <a href="https://arxiv.org/abs/1706.03762">"Attention Is All You Need"</a></em></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
</div>

## 📋 Features

- **Complete Implementation**: Full transformer architecture with encoder and decoder
- **Modular Design**: Easily adaptable components for your own NLP tasks
- **Efficient Attention**: Multi-head attention implementation with masking
- **Detailed Documentation**: Comments and design explanations throughout the code
- **Training Pipeline**: Ready-to-use training and inference scripts
- **Example Application**: English to Italian machine translation

## 🧠 The Transformer Architecture Explained

<div align="center">
  <img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png" alt="Transformer Block" width="450px">
</div>

The Transformer revolutionized NLP by replacing recurrence and convolutions with **self-attention mechanisms** that capture dependencies regardless of their distance in the sequence.

### ⚡ Self-Attention Mechanism

<div align="center">
  <img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" alt="Self-Attention Calculation" width="500px">

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```
</div>

Self-attention computes a weighted sum of all positions in a sequence, with weights determined by the compatibility of query-key pairs.

### 🔄 Multi-Head Attention

<div align="center">
  <img src="https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png" alt="Multi-Head Attention" width="550px">
</div>

Instead of performing a single attention function, the transformer uses multiple attention heads to capture different types of dependencies:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
where headᵢ = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 📍 Positional Encoding

<div align="center">
  <img src="https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png" alt="Positional Encoding" width="400px">
</div>

Since the transformer has no recurrence, it needs positional encodings to make use of sequence order:

```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
```

## 🛠️ Components

| File | Description |
|------|-------------|
| `model.py` | Core transformer architecture implementation |
| `train.py` | Training loop and optimization procedures |
| `translate.py` | Inference and translation functionality |
| `dataset.py` | Data loading and preprocessing utilities |
| `config.py` | Configuration parameters and hyperparameters |

## 💻 Quick Start

```bash
# Clone the repository
git clone https://github.com/bikrammajhi/Transformer-from-scratch-using-PyTorch.git
cd Transformer-from-scratch-using-PyTorch

# Install requirements
pip install -r requirements.txt

# Train the model
python train.py

# Translate a sentence
python translate.py
```

## 🔍 Architecture Overview

<div align="center">
  <img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="Encoder-Decoder" width="500px">
</div>

```
Transformer
├── Encoder (6 layers)
│   ├── Multi-Head Attention
│   ├── Feed Forward Network
│   └── Layer Normalization
└── Decoder (6 layers)
    ├── Masked Multi-Head Attention
    ├── Multi-Head Attention
    ├── Feed Forward Network
    └── Layer Normalization
```

## 📊 Training Results

### Training Log Example

```
Processing Epoch 00: 100% 3638/3638 [23:45<00:00, 2.55it/s, loss=6.048]
Processing Epoch 01: 100% 3638/3638 [23:47<00:00, 2.55it/s, loss=5.207]
Processing Epoch 02: 100% 3638/3638 [23:47<00:00, 2.55it/s, loss=4.183]
```

### Translation Example

```
Using device: cpu
    SOURCE: I am not a very good a student.
 PREDICTED: Io non ho il il. ⏎
```

## 🔄 Using Pre-trained Weights

```bash
# Create necessary directories
mkdir -p opus_books_weights

# Download pre-trained weights and tokenizer files
# Will update the instruction here, when weights upload finishes
```

## 📈 Implementation Details

Our implementation follows the original paper specifications:

| Parameter | Value |
|-----------|-------|
| Encoder/Decoder Layers | 6 each |
| Attention Heads | 8 |
| Embedding Dimension | 512 |
| Feed-forward Dimension | 2048 |
| Dropout Rate | 0.1 |

## 📚 Training Data

The model is trained on the [Opus Books dataset](https://opus.nlpl.eu/Books.php) from Hugging Face, a parallel corpus of various books translated across multiple languages.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

MIT License - See [LICENSE](LICENSE) for more information.

## ⭐ Show Your Support

If you find this implementation helpful, please give it a star! It helps the project gain visibility and encourages further development.

## 🙏 Acknowledgements

- [Umar Jamil](https://www.youtube.com/watch?v=ISNdQcPhsts) for his video on transformer implementation from scratch
- Campusx and CodeEmporium for their educational content on transformers
- The authors of the original [Transformer paper](https://arxiv.org/abs/1706.03762)
- [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) for his illustrated guide to transformers
