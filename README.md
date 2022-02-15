# Retrieval Augmentation for Deep Neural Networks

Code for the paper [Retrieval Augmentation for Deep Neural Networks](https://ieeexplore.ieee.org/document/9533978)

# Introduction

The most common methodology in deep learning involves the supervised training of a neural network with input-output pairs, so as to minimize a given loss function. In general, deep neural networks predict the output conditioned solely on the current input or, more recently, leveraging an attention mechanism that focuses only on parts of the input as well. This leaves the rest of the labeled examples unused for the current prediction, either during training or inference.

In this work, we leverage similar examples in the training set to improve the performance and interpretability of deep neural networks, both at training and testing time. We propose an approach that retrieves the nearest training example to the one being processed and uses the corresponding target example (i) as auxiliary context to the input (e.g. combining the input together with the retrieved target), or (ii) to guide the attention mechanism of the neural network.

We show that the retrieved target can be easily incorporated in an LSTM model, making use of its initial memory state. We also present a new multi-level attention method that attends to the inputs and to the target of the nearest example.

We evaluate the proposed approach on image captioning and sentiment analysis. In brief, image captioning involves generating a textual description of an image. The dominant framework involves using a CNN as an encoder to represent the image, and passes this representation to a RNN decoder that generates the respective caption, combined with neural-attention. In turn, sentiment analysis aims to classify the sentiment of an input text. Within neural methods, RNNs and CNNs are commonly used for sentiment analysis, recently also combining attention mechanisms.

Our general aim is to demonstrate the effectiveness of the proposed approach, which can also be easily integrated in other neural models, by applying it to standard methods for two different tasks (i.e., generation and classification) and by using a retrieval mechanism with different modalities (i.e., image and text retrieval).

# Citation

```
@inproceedings{ramos2021retrieval,
  title={Retrieval Augmentation for Deep Neural Networks},
  author={Ramos, Rita Parada and Pereira, Patr{\'\i}cia and Moniz, Helena and Carvalho, Joao Paulo and Martins, Bruno},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}

```

# Baselines

Our Retrieval Augmentation approach was implemented in the following [Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) baseline and [Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb) baseline.
