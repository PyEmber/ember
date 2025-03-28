# Ember Data Processing - Quickstart Guide

This guide introduces Ember's data processing system, which provides tools for loading, transforming, and evaluating data across various benchmarks and datasets.

## 1. Introduction to Ember Data

Ember's data module provides:
- **Standardized Dataset Loading**: Unified access to common benchmarks and custom datasets
- **Flexible Transformations**: Pipelines for preprocessing and normalizing data
- **Evaluation Framework**: Tools for measuring model performance across tasks
- **Sampling Controls**: Methods for dataset subsampling and stratification
- **Data Registry**: Registry of popular evaluation benchmarks with metadata

## 2. Loading Standard Datasets

```python
from ember.core.utils.data.service import DataService
from ember.core.utils.data.base.config import DatasetConfig

# Initialize data service
data_service = DataService()

# Load a standard benchmark dataset
mmlu_data = data_service.load_dataset(
    dataset_name="mmlu",
    subset="high_school_biology",
    split="test",
    limit=100  # Optional limit
)

# Access the data
for item in mmlu_data:
    print(f"Question: {item.question}")
    print(f"Choices: {item.choices}")
    print(f"Answer: {item.answer}")
    print("---")
```

## 3. Using Dataset Transformers

```python
from ember.core.utils.data.base.transformers import (
    ShuffleChoicesTransformer,
    ContextEnricherTransformer,
    TextCleanerTransformer
)

# Create transformers
shuffle_transformer = ShuffleChoicesTransformer(seed=42)
cleaner_transformer = TextCleanerTransformer()
context_transformer = ContextEnricherTransformer(
    context_retriever=my_retriever_function
)

# Apply transformations sequentially
transformed_data = data_service.load_dataset(
    dataset_name="mmlu",
    subset="high_school_biology",
    transformers=[
        cleaner_transformer,
        shuffle_transformer,
        context_transformer
    ]
)

# Access transformed data
for item in transformed_data:
    print(f"Question: {item.question}")
    print(f"Context: {item.context}")  # Added by the context transformer
    print(f"Shuffled Choices: {item.choices}")  # Shuffled by the transformer
```

## 4. Creating Custom Datasets

```python
from ember.core.utils.data.base.models import DataItem, Dataset
from typing import List, Dict, Any
import json

# Define a function to load custom data
def load_my_dataset() -> Dataset:
    # Load data from a file
    with open("my_dataset.json", "r") as f:
        data = json.load(f)
    
    # Convert to DataItem objects
    items: List[DataItem] = []
    for entry in data:
        item = DataItem(
            id=entry["id"],
            question=entry["question"],
            choices=entry["options"],
            answer=entry["correct_option"],
            metadata={
                "category": entry["category"],
                "difficulty": entry["difficulty"]
            }
        )
        items.append(item)
    
    # Create and return dataset
    return Dataset(
        name="my_custom_dataset",
        items=items,
        metadata={"source": "custom", "version": "1.0"}
    )

# Register custom dataset loader
data_service.register_loader("my_dataset", load_my_dataset)

# Use it like any other dataset
my_data = data_service.load_dataset("my_dataset")
```

## 5. Sampling and Filtering

```python
from ember.core.utils.data.base.samplers import (
    RandomSampler,
    StratifiedSampler,
    FilterSampler
)

# Create samplers
random_sampler = RandomSampler(n=50, seed=42)
stratified_sampler = StratifiedSampler(
    n=50,
    stratify_by="metadata.category",
    seed=42
)
filter_sampler = FilterSampler(
    filter_fn=lambda item: item.metadata.get("difficulty") == "hard"
)

# Use with dataset loading
hard_items = data_service.load_dataset(
    dataset_name="mmlu",
    sampler=filter_sampler
)

# Or apply to existing dataset
random_subset = random_sampler.sample(my_dataset)
stratified_subset = stratified_sampler.sample(my_dataset)

print(f"Original size: {len(my_dataset)}")
print(f"Random sample size: {len(random_subset)}")
print(f"Stratified sample size: {len(stratified_subset)}")
```

## 6. Evaluating Model Performance

```python
from ember.core.utils.eval.pipeline import EvaluationPipeline
from ember.core.utils.eval.evaluators import (
    ExactMatchEvaluator,
    F1ScoreEvaluator,
    MultipleChoiceEvaluator
)
from ember.core.registry.model.base.services import ModelService

# Initialize model service
model_service = ModelService()
model = model_service.get_model("openai:gpt-4o")

# Create evaluators
multiple_choice_evaluator = MultipleChoiceEvaluator()
exact_match_evaluator = ExactMatchEvaluator()

# Create evaluation pipeline
eval_pipeline = EvaluationPipeline(
    dataset=mmlu_data,
    evaluators=[multiple_choice_evaluator],
    model=model
)

# Run evaluation
results = eval_pipeline.evaluate()

# Print results
print(f"Accuracy: {results.metrics['accuracy']:.2f}")
print(f"Per-category breakdown: {results.metrics['category_accuracy']}")
```

## 7. Working with Evaluation Results

```python
from ember.core.utils.eval.pipeline import EvaluationResults
import matplotlib.pyplot as plt

# Assuming we have evaluation results
# results: EvaluationResults

# Access overall metrics
accuracy = results.metrics["accuracy"]
f1 = results.metrics.get("f1_score", 0.0)

# Access per-item results
for item_result in results.item_results:
    item_id = item_result.item_id
    correct = item_result.correct
    model_answer = item_result.model_output
    expected = item_result.expected_output
    
    if not correct:
        print(f"Item {item_id} was incorrect:")
        print(f"  Expected: {expected}")
        print(f"  Model output: {model_answer}")

# Plot results
categories = results.metrics["category_accuracy"].keys()
accuracies = [results.metrics["category_accuracy"][cat] for cat in categories]

plt.figure(figsize=(10, 6))
plt.bar(categories, accuracies)
plt.title("Accuracy by Category")
plt.xlabel("Category")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("category_results.png")
```

## 8. Built-in Datasets

Ember provides ready-to-use loaders for popular benchmarks:

```python
# MMLU (Massive Multitask Language Understanding)
mmlu = data_service.load_dataset(
    dataset_name="mmlu",
    subset="high_school_mathematics"
)

# TruthfulQA
truthful_qa = data_service.load_dataset(
    dataset_name="truthful_qa",
    subset="generation"
)

# HaluEval (Hallucination Evaluation)
halu_eval = data_service.load_dataset(
    dataset_name="halueval",
    subset="knowledge"
)

# CommonsenseQA
commonsense_qa = data_service.load_dataset(
    dataset_name="commonsense_qa",
    split="validation"
)

# Short Answer QA
short_answer = data_service.load_dataset(
    dataset_name="short_answer",
    subset="factual"
)
```

## 9. Best Practices

1. **Cache Datasets**: Use the caching features to avoid reloading
2. **Apply Transformers Carefully**: Consider the order of transformations
3. **Stratified Sampling**: Ensure representative subsets when sampling
4. **Multiple Evaluators**: Use multiple evaluators for comprehensive assessment
5. **Metadata**: Add rich metadata to custom datasets for better filtering
6. **Batch Processing**: Process data in batches for large datasets

## Next Steps

Learn more about:
- [Model Registry](model_registry.md) - Managing LLM configurations
- [Operators](operators.md) - Building computational units
- [Evaluation Metrics](../advanced/evaluation_metrics.md) - Detailed metrics and evaluation approaches
- [Custom Transformers](../advanced/custom_transformers.md) - Building custom data transformations