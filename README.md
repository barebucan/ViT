# Vision Transformer (ViT) on ImageNet100

This repository contains the implementation of a Vision Transformer (ViT) model trained on the ImageNet100 dataset. The project aims to demonstrate the effectiveness of transformer architectures in computer vision tasks.

## Project Structure

```
.
├── __pycache__/
├── config.py
├── dataset.py
├── main.py
└── net.py
```

## File Descriptions

- `__pycache__/`: Python bytecode cache
- `config.py`: Configuration settings for the ViT model and training process
- `dataset.py`: Dataset loading and preprocessing for ImageNet100
- `main.py`: Main script for training and evaluating the ViT model
- `net.py`: Implementation of the Vision Transformer architecture

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/vit-imagenet100.git
   cd vit-imagenet100
   ```

2. Install the required dependencies:
   ```
   pip install torch torchvision transformers pandas numpy pillow
   ```

3. Download the ImageNet100 dataset and update the data path in `config.py`.

## Usage

To train the Vision Transformer on ImageNet100:

```
python main.py
```

Modify the hyperparameters and model configuration in `config.py` as needed.

## Model Architecture

This project implements a Vision Transformer with the following key components:

- Patch embedding
- Positional encoding
- Multi-head self-attention layers
- Feed-forward networks
- Layer normalization

The specific architecture details can be found in `net.py`.

## Results

(Add information about the model's performance, accuracy, and any visualizations once training is complete.)

## Contributing

Contributions to improve the model architecture, training process, or documentation are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Vision Transformer architecture is based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.
- ImageNet100 is a subset of the ImageNet dataset.
