# Visual Question Answering
## Team Members
Hongyu <br> 
Jaineet 

## Motivation - 
To answer questions based on input images accurately.
To create a model with an feasible architecture.
To create an attention layer to improve the overall accuracy.
Dataset Used - CLEVR Dataset. It consists of 70,000 training images and 699,989 questions. 

## Sources 
https://www.kaggle.com/code/bhavikardeshna/visual-question-answering-multimodal-transfomer 
https://www.kaggle.com/code/marcelosabaris/visualquestionanswering
https://cs.stanford.edu/people/jcjohns/clevr/ 

## Methodology 
#### Training Dataset - 20,000 images
#### Validation Dataset - 5000 images
#### Number of Epochs - 10
#### Solved as a classification problem for 1-word answers (99 tokens/labels).
#### Hardware - Free GPU (Tesla T4) on google colab. 
#### Software - Python 3.9.16, TensorFlow 2.12.0
#### Tokens of length 50 (using padding) encoded using tfds (tensorflow datasets) token text encoder. 
#### Images of size (200,200,3) and (224,224,3) used in different models. 
#### Loss function - sparse categorical cross entropy, optimizer - Adam
#### Batch Size - 50.
#### Learning rate - 0.001 with decay.
#### All models are run on tf.random.set_seed(1). 

### Model 1 Spec (MobileNetV2 (pre-trained on Imagenet) + Bi-Directional LSTM): 
pre-trained MobileNetV2, three bi-directional LSTM layers with 256,256 and 512 state units, 
concat layer for combining image and text features, softmax layer. 
Total parameters - 9,335,715 , Non-trainable - 34,112. Average Training Time per Epoch - 190s.

### Model 2 Spec (VGG-16 Encoder + Transformer Encoder (MultiAttentionHead Layer): 
Vgg-16 encoder -> Conv layers 64 to 512, transformer encoder -> 5 layers of (MultiHeadAttention (3) + LayerNorm (previous + current)+ Dense ), 
concat layer for combining image and text features, softmax layer. 
Total parameters - 15,058,405. Average Training Time per Epoch - 290s.

### Model 3 Spec (VGG-16 Encoder + Bi-Directional LSTM): 
Vgg-16 encoder -> Conv layers 64 to 512, three bi-directional LSTM layers with 256,256 and 512 state units,
concat layer for combining image and text features, softmax layer. 
Total parameters - 21,716,387. Average Training Time per Epoch - 310s.

### Model 4 Spec (VGG-16 Encoder + LSTM +  Attention Mechanism): 
Vgg-16 encoder -> Conv layers 64 to 512, one LSTM layer with 256 state units, Attention Layer :  [Image features to dense layer (256 units),
softmax layer (1 unit) (attention weights), multiplying question features with attention weights, 
Lambda layer to compute the sum the weights dimension of attention weights layer, 
concat layer for combining image features and summed attention weights, softmax layer. 
Total parameters - 24,197,540. Average Training Time per Epoch - 290s.

### Model 5 Spec (MobileNetV2 (pre-trained on Imagenet) + Bi-Directional LSTM Version-2): 
Vgg-16 encoder -> Conv layers 64 to 512, three bi-directional LSTM layers with 256, two Dense layers with 512 and 256 units,
concat layer for combining image and text features, softmax layer.
Total parameters - 10,444,195, Non-trainable - 34,112. Average Training Time per Epoch - 200s.

### Future Objectives:
Train all the models for more epochs to find the true performance. 
Fine-Tune the existing models to increase performance.
Focus on creating more robust techniques for combining image and text features.
Test the performance using Glove or other pre-trained text embeddings instead of tfds text encoder.  
Test models on different datasets (DAQAUR or VQA).

### For more details, kindly refer to the technical report pdf. 










