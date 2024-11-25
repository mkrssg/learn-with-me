"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MatryoshkaLoss using MultipleNegativesRankingLoss. This trains a model at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset at the different output dimensions.

Usage:
python matryoshka_nli.py

OR
python matryoshka_nli.py pretrained_transformer_model_name
"""

import logging
import sys

from datetime import datetime

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 1
matryoshka_dims = [768, 512, 256, 128, 64]

# Save path of the model
output_dir = f"output/matryoshka_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info(train_dataset)
#train_dataset = train_dataset[:10000]
logging.info(eval_dataset)

# If you wish, you can limit the number of training samples
train_dataset = train_dataset.select(range(5000))

# 3. Define our training loss
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval_dataset["sentence1"],
            sentences2=stsb_eval_dataset["sentence2"],
            scores=stsb_eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
    )
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="matryoshka-nli",  # Will be used in W&B if `wandb` is installed
)
print(type(train_dataset))

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

#### Results:
# 2024-09-26 09:32:18 - EmbeddingSimilarityEvaluator: Evaluating the model on the sts-test-768 dataset (truncated to 768):
# 2024-09-26 09:32:34 - Cosine-Similarity :       Pearson: 0.7332 Spearman: 0.7320
# 2024-09-26 09:32:34 - Manhattan-Distance:       Pearson: 0.7332 Spearman: 0.7157
# 2024-09-26 09:32:34 - Euclidean-Distance:       Pearson: 0.7350 Spearman: 0.7169
# 2024-09-26 09:32:34 - Dot-Product-Similarity:   Pearson: 0.4186 Spearman: 0.4135
# 2024-09-26 09:32:34 - EmbeddingSimilarityEvaluator: Evaluating the model on the sts-test-512 dataset (truncated to 512):
# 2024-09-26 09:32:41 - Cosine-Similarity :       Pearson: 0.7398 Spearman: 0.7268
# 2024-09-26 09:32:41 - Manhattan-Distance:       Pearson: 0.7346 Spearman: 0.7163
# 2024-09-26 09:32:41 - Euclidean-Distance:       Pearson: 0.7359 Spearman: 0.7168
# 2024-09-26 09:32:41 - Dot-Product-Similarity:   Pearson: 0.4933 Spearman: 0.4944
# 2024-09-26 09:32:41 - EmbeddingSimilarityEvaluator: Evaluating the model on the sts-test-256 dataset (truncated to 256):
# 2024-09-26 09:32:48 - Cosine-Similarity :       Pearson: 0.7314 Spearman: 0.7247
# 2024-09-26 09:32:48 - Manhattan-Distance:       Pearson: 0.7326 Spearman: 0.7151
# 2024-09-26 09:32:48 - Euclidean-Distance:       Pearson: 0.7346 Spearman: 0.7163
# 2024-09-26 09:32:48 - Dot-Product-Similarity:   Pearson: 0.4963 Spearman: 0.4919
# 2024-09-26 09:32:48 - EmbeddingSimilarityEvaluator: Evaluating the model on the sts-test-128 dataset (truncated to 128):
# 2024-09-26 09:32:54 - Cosine-Similarity :       Pearson: 0.7107 Spearman: 0.7130
# 2024-09-26 09:32:54 - Manhattan-Distance:       Pearson: 0.7274 Spearman: 0.7100
# 2024-09-26 09:32:54 - Euclidean-Distance:       Pearson: 0.7290 Spearman: 0.7110
# 2024-09-26 09:32:54 - Dot-Product-Similarity:   Pearson: 0.4567 Spearman: 0.4354
# 2024-09-26 09:32:54 - EmbeddingSimilarityEvaluator: Evaluating the model on the sts-test-64 dataset (truncated to 64):
# 2024-09-26 09:33:01 - Cosine-Similarity :       Pearson: 0.7062 Spearman: 0.7067
# 2024-09-26 09:33:01 - Manhattan-Distance:       Pearson: 0.7185 Spearman: 0.7031
# 2024-09-26 09:33:01 - Euclidean-Distance:       Pearson: 0.7219 Spearman: 0.7065
# 2024-09-26 09:33:01 - Dot-Product-Similarity:   Pearson: 0.3273 Spearman: 0.3327