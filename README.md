# Transformer Implementation from Scratch 🚀

A PyTorch implementation of the Transformer architecture as described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). This project includes a complete, modular implementation of the Transformer machine translation tasks from english to italian.

## 🌟 Features

- Complete transformer architecture implementation
- Modular design with separate encoder and decoder components
- Multi-head attention mechanism
- Support for custom tokenization
- Training and inference scripts included
- Translation example implementation

## 🛠️ Components

- `model.py`: Core transformer architecture
- `train.py`: Training loop and utilities
- `translate.py`: Inference and translation script
- `dataset.py`: Data loading and preprocessing
- `config.py`: Configuration and hyperparameters

## 🚀 Quick Start

```bash
# Clone the repository
git https://github.com/bikrammajhi/Transformer-from-scratch-using-PyTorch.git
cd Transformer-from-scratch-using-PyTorch

# Install requirements
pip install -r requirements.txt

# Train the model
python train.py

# Translate a sentence
python translate.py

```
or
## Load pre-trained weights 

### Create necessary directories
```bash

mkdir -p opus_books_weights

## Download pre-trained weights and tokenizer files
- will update the instruction here, when weights upload finishes
````


## 📋 Model Architecture

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

### Results
Training
```bash
Processing Epoch 00: 100% 3638/3638 [23:45<00:00,  2.55it/s, loss=6.048]
Processing Epoch 01: 100% 3638/3638 [23:47<00:00,  2.55it/s, loss=5.207]
Processing Epoch 02: 100% 3638/3638 [23:47<00:00,  2.55it/s, loss=4.183]
```
Machine translation
```bash
Using device: cpu
    SOURCE: I am not a very good a student.
 PREDICTED: Io non ho il  il  .  ⏎  
```
## 📚 Training Data

The model can be trained on any parallel corpus. The example implementation uses the [Opus Books dataset](https://opus.nlpl.eu/Books.php) from huggingface.

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License

MIT License - feel free to use this code for your own projects!

## ⭐️ Show Your Support

If you find this implementation helpful, give it a star! ⭐️

## Special Thanks
- Umar Jamil for his video on transformer from Scratch [video](https://www.youtube.com/watch?v=ISNdQcPhsts)
- Campusx and CodeEmporium for helping me understand transformer
