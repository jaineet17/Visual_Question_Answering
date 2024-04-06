# Visual Question Answering

## Team Members

- Hongyu
- Jaineet

## Motivation

Our project aims to accurately answer questions based on input images by developing a model with a feasible architecture. We focus on creating an attention layer to improve the model's overall accuracy. We utilize the CLEVR Dataset, which consists of 70,000 training images and 699,989 questions, for our training and validation purposes.

## Sources

- [Visual Question Answering Multimodal Transformer on Kaggle](https://www.kaggle.com/code/bhavikardeshna/visual-question-answering-multimodal-transfomer)
- [VisualQuestionAnswering on Kaggle](https://www.kaggle.com/code/marcelosabaris/visualquestionanswering)
- [CLEVR Dataset by Stanford University](https://cs.stanford.edu/people/jcjohns/clevr/)

## Methodology

- **Training Dataset**: 20,000 images
- **Validation Dataset**: 5,000 images
- **Number of Epochs**: 10
- **Problem Approach**: Treated as a classification problem for 1-word answers, with 99 tokens/labels.
- **Hardware**: Free GPU (Tesla T4) on Google Colab.
- **Software**: Python 3.9.16, TensorFlow 2.12.0
- **Tokens**: Length of 50 (using padding), encoded using TensorFlow Datasets (tfds) token text encoder.
- **Image Sizes**: (200,200,3) and (224,224,3) used in different models.
- **Loss Function**: Sparse categorical cross-entropy
- **Optimizer**: Adam
- **Batch Size**: 50
- **Learning Rate**: 0.001 with decay
- **Random Seed**: All models are run on `tf.random.set_seed(1)`.

### Models Overview

#### Model 1: MobileNetV2 + Bi-Directional LSTM

- **Architecture**: Pre-trained MobileNetV2, three bi-directional LSTM layers (256, 256, and 512 state units), concat layer, softmax layer.
- **Parameters**: 9,335,715 (Non-trainable: 34,112)
- **Avg. Training Time/Epoch**: 190s

#### Model 2: VGG-16 Encoder + Transformer Encoder

- **Architecture**: VGG-16 encoder, transformer encoder (5 layers of MultiHeadAttention (3) + LayerNorm + Dense), concat layer, softmax layer.
- **Parameters**: 15,058,405
- **Avg. Training Time/Epoch**: 290s

#### Model 3: VGG-16 Encoder + Bi-Directional LSTM

- **Architecture**: VGG-16 encoder, three bi-directional LSTM layers (256, 256, and 512 state units), concat layer, softmax layer.
- **Parameters**: 21,716,387
- **Avg. Training Time/Epoch**: 310s

#### Model 4: VGG-16 Encoder + LSTM + Attention Mechanism

- **Architecture**: VGG-16 encoder, one LSTM layer (256 state units), Attention Layer, concat layer, softmax layer.
- **Parameters**: 24,197,540
- **Avg. Training Time/Epoch**: 290s

#### Model 5: MobileNetV2 + Bi-Directional LSTM V2

- **Architecture**: VGG-16 encoder, three bi-directional LSTM layers (256 state units), two Dense layers (512 and 256 units), concat layer, softmax layer.
- **Parameters**: 10,444,195 (Non-trainable: 34,112)
- **Avg. Training Time/Epoch**: 200s

## Future Objectives

- Train all models for more epochs to better assess performance.
- Fine-tune existing models to increase performance.
- Develop more robust techniques for combining image and text features.
- Explore the performance impact of using GloVe or other pre-trained text embeddings instead of tfds text encoder.
- Test models on different datasets (DAQUAR or VQA).

For more detailed insights into our project, please refer to our technical report PDF.
