# Matryoshka Embeddings

Recent embedding models generate embeddings of very high dimentionality, allowing better quality and performance but coming with less efficiency in downstream tasks such as in search applications or RAG. 

To tackle this issue Kusupati et al. (2022) propose embedding models which produce embeddings which can be shrunk without loosing too much information and performance.

## How does it work?

Matryoshka embedding models can create useful embeddings of a variety of dimensions. They are trained so that the small truncated embeddings are still useful. Like a Matryoshka doll the levels are nested in one embedding. Depending on your application you could require less information so that you can use a embedding with fewer dimensions meaning less details. In case you need more information you could later switch to a embedding level with more details resulting in better quality traded for more ressource requirements.

The models are trained so that the more important information is stored in earlier dimensions of the vector, the less important information at the end of the embedding can be truncated without loosing too much performance on downstream tasks.

## Use case

- Valuable for scaling search applications where efficient retrieval is required such as large RAG apps. You can use the truncated embeddings to find the best k matches, afterwards you can use the full dimensionality again to rerank with higher quality. 
- This allows less required memory, costs and faster search since e.g. a similarity calculation is performed more efficiently.


## How are the Matryoshka embeddings trained?

- NN produces embeddings for the training batch
- Use loss function to calculate loss value of full size embeddings and the loss values of embeddings of different smaller dimensionalities (e.g. 768, 512, 256, 128, and 64)
- Sum all the loss values together to a final loss value
- Use the optimizer to adjust the model weights so that the final loss value is reduced

-> This causes the model to place important information at the beginning of the embedding so that truncation will keep the quality of the embedding


### Ressources 
- https://arxiv.org/abs/2205.13147
- https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss
- https://huggingface.co/blog/matryoshka
- https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/matryoshka
