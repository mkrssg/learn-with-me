# Quantization

Quantization is a technique to reduce computation and memory costs by reducing weights and activations with low precision data types such as 8 bit integer instead of 32 bit floats. 

Quantization is used within a neural network and allows the model to 
- consume less memory 
- operate faster since operations like the matrix multiplication can be performed faster

Common quantization cases:
- float32 -> float16
- float32 -> int8


# Vector Quantization

Besides optimizing the interal workflow in a Neural Network through quantization where you reduce the precision of weights, also the resulting embeddings can be quantitized which is called vector quantization. This allows a more efficient process for bigger applications such as RAG applications. 

Since RAG (Retrieval Augmented Generation) applications are getting widely spread in the industries new concepts arise to improve their performance and costs. 

Vector Quantization focuses on the R step in RAG, the Retrieval step of the semantic search. 

Vector Quantization claims to:
- reduce the costs for storing embeddings as it reduces disk storage
- reduce memory usage 
- increase retrieval speed
- while maintaining good quality in the retrieval step 

## Why Vector Quantization?

Production use cases using for example a semantic search backbone such as a RAG application may have difficulties to scale leading to high expenses and high latency affecting the user experience negatively.


## Methods

There are different approaches when it comes to scaling embeddings. Classical approaches are dimensionality reduction algorithms, which basically aim to reduce the dimensions of the embeddings while maintaining the meaningful features allowing them to be more compact without loosing relevant information. Typical dimensionality reduction algorithms are PCA or UMAP. Nevertheless, those approaches tend to reduce perfomance in downstream tasks using the resulting vectors.
 
Another recent approach is Matryoshka Representations (see file). 

Further, there is quantization, another technique which does not reduce the dimensionality of the vector but reduce the size of each value in the vector. 

### Binary Quantization

Binary quantization aims to reduce the float32 values in an embeddin to 1-bit values which results in a 32x reduction in memory and storage.

f(x) =
- 0 if x <= 0
- 1 if x > 0

while reducing the storage by 32x, the total retrieval performance is preserved with ~92.5%

Yamada et al. (2021) propose a rerank step, to boost the retrieval performance to 96%. First k documents will be retrieved using the binary embedding and in a second step rerank this list of k documents using the original float32 embedding.


### Scalar Quantization

Maps each value of the original embedding which is usually from type floating point to a integer. 
There are different approaches of scalar quantization:
- 8-bit quantization: Splits the range of the float values into 255 buckets leading to a storage reduction by 4 as each dimension only stores 1 byte and not 4 when working with 32-bit floats.
- 1-bit quantization: a more radical appraoch where the floating point range is cut in half leading to 0 or 1 as a outcome value. 

### Product Quantization

A similar but more complex approach where vectors are split across dimensions into subvectors which are run through a clustering algorithm. This allows drastic reduction in data storage while also loosing accuracy. 


## Examples

see `02_vector_quantization.ipynb`

## Sources
- https://neuml.hashnode.dev/all-about-vector-quantization
- https://huggingface.co/blog/embedding-quantization
- https://huggingface.co/docs/optimum/en/concept_guides/quantization
- https://qdrant.tech/articles/binary-quantization/#