# text2rdf

This project implements a model architecture, including training and testing functionality, to summarize input sentences 
in terms of RDF triples. 

The model architecture consists of three parts:

1. Encoder
2. Soft Attention Module
3. Decoder

These parts can further be described as follows:

## Encoder
The Encoder consists of a BERT language model and encodes input sentences, returning one so-called annotation vector per 
input token passed into the Encoder. Tokens are derived from sentences by means of the tokenizer originally used for 
training the BERT model. All Encoder-related parts are provided by the [Transformers](https://huggingface.co/transformers/index.html) 
library. 

## Soft Attention Module
This [Soft Attention module](http://proceedings.mlr.press/v37/xuc15.pdf) computes one attention weight per annotation vector produced by the Encoder, where an attention weight 
expresses the importance of one annotation vector relative to the other annotation vectors given the Decoder's previous 
hidden state. 

## Decoder
The Decoder consists of a Recurrent Neural Network (RNN), composed of Gated Recurrent Units, followed by a fully-connected 
layer. First, the Decoder takes the annotation vectors provided by the Encoder and the attention weights produced by the 
Soft Attention module and produces a weighted average of the annotation vectors, where the weights are the attention 
weights. The resulting weighted average is called *context vector*. The context vector is passed through both the RNN and 
the fully-connected layer (with non-linearities in between) and produces a probability distribution over the output 
vocabulary. The token associated with the highest probability is then chosen in the given time step as the predicted output word 
token. 

## Dataset
The system has been trained using the [WebNLG challenge 2020](https://webnlg-challenge.loria.fr/challenge_2020/) dataset. 

## Training the system
For training the system, place the unzipped dataset folder `WebNLG` into the folder where also this README is located and 
run `python3 main.py X Y Z` (also in the folder where this README is located), where `X` stands for the number of training 
epochs, `Y` specifies the the minimal number of triples to be predicted per input sentence, and `Z` 
refers to the maximal number of target triples per input sentence that a system is supposed to be trained on. 

An object containing evaluation statistics collected both throughout and after the end of training is saved at the end of 
training. Examples of this, as well as recorded indicative print-statements collected for some trained models throughout training, can be found in the folder `train_results`.

# Further Information
For more information on this project, please be referred to the [attached project report](https://github.com/Bick95/text2rdf/blob/main/text2rdf_Semantic_Web_Technology_Project_Report.pdf). 
