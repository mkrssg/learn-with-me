# Embeddings

## What are embeddings and why do we use them?

Embeddings are a numerical representation of complex objects which can be images, audio, text, time series etc. Those objects can be encoded using different techniques such as one-hot encodings, word-2-vec or modern neural network architectures such as transformer models. 

The embedding or the vector representation of the complex object allows the computer to work with it based on mathematically concepts. For example we can calculate the similarity of two objects (which are e.g. two different text paragraphs) by calculating e.g. the cosine similarity between the two vector representations of those objects. 

Using this concept, we can for example build recommendation systems which recommend a user another movie based on the one he liked before. For that a set of movies is encoded and the most similar vector to the one he liked is calculated allowing us to recommend another similar one of the set of movies.

## How is a embedding model trained (neural network)?

Basic neural network training procedure:
- NN generates embeddings for your training batch (e.g. texts)
- Use a loss function to calculate the loss value representing the quality of the embedding
- Use an optimizer to adjust the model weights during the training to reduce the loss value