# Deep Neural Network Speech Recognition 

In this project we built a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline. The full pipeline is summarized in the figure below.

<br />


![DNN Architecture](https://github.com/3ba2ii/DNN-Speech-Recognition/raw/5c737dced0e893a5910ade9b49651377e725766e/images/pipeline.png)


<br />

## Content

- [Deep Neural Network Speech Recognition](#deep-neural-network-speech-recognition)
  - [Content](#content)
  - [Description](#description)
  - [What To Improve](#what-to-improve)
        - [Methods to decrease the error :](#methods-to-decrease-the-error-)
    - [Prerequisites](#prerequisites)
        - [Install Keras using pip](#install-keras-using-pip)
        - [Install Keras using conda](#install-keras-using-conda)
  - [Network Architecture](#network-architecture)
  - [Optimizer and Loss Function](#optimizer-and-loss-function)
  - [Authors](#authors)
  - [Contributing](#contributing)

## Description 

The pipeline will accept raw audio as input and make a pre-processing step that converts raw audio to one of two feature representations that are commonly used for ASR ([Spectrogram](https://en.wikipedia.org/wiki/Spectrogram) or [MFCCs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)) in this project we've used a Convolutional Layer to extract features. Then these features are fed into an acoustic model which accepts audio features as input and returns a probability distribution over all potential transcriptions. The last step is that the pipeline takes the output from the acoustic model and returns a predicted transcription.

<br />


## What To Improve 

We should be able to get better performance on both training and validation set.

##### Methods to decrease the error :

  
 - [ ] Try getting larger dataset.
 - [ ] Try adding language model after the acoustic model.
 - [ ] Try training for more epochs >20.
 - [ ] Try deeper neural network or pre-trained network.
 - [ ] Try using another type of RNNs like LSTM, or GRU



<br />

### Prerequisites

This project uses [keras framework](https://keras.io/getting_started/) follow the commands below to install it appropriately 

##### Install Keras using pip 
```bash
pip install Keras
```
##### Install Keras using conda 
```bash
conda install -c conda-forge keras
```
<br />

## Network Architecture 

used a 1D convolutional layer to extract features and added [BatchNormalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) layer after each layer to speed up learning process, a dropout layers to prevent the model from overfitting then used a combination of ```Bidirectional + SimpleRNNs```; the reason why i chose SimpleRNNs as it was so fast compared to GRU and LSTM. <br/>
The output of the acoustic model is connected to a softmax function to predict the probability of transcriptions.


![Network Architecture](https://i.ibb.co/H2dFPJZ/Screen-Shot-2020-06-20-at-2-30-12-PM.png)

> feel free to take a look at final model in ```sample_models.py``` 



<br />

## Optimizer and Loss Function
we trained the acoustic model with the [CTC loss](https://machinelearning-blog.com/2018/09/05/753/) along with SGD optimizer with learning rate 0.02.

```python 

def add_ctc_loss(input_to_softmax):

    the_labels    = Input(name='the_labels',
                          shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length',
                          shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length',
                          shape=(1,), dtype='int64')

    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)

    # CTC loss is implemented in a lambda layer

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])

    model = Model(
        inputs=[input_to_softmax.input, the_labels,
                input_lengths, label_lengths],
        outputs=loss_out)

    return model


```


## Authors

- **Ahmed Abd-Elbakey Ghonem** - [**Github**](https://github.com/3ba2ii)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.



