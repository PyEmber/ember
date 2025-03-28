# Using Specialized Datasets in Ember

This guide demonstrates how to use Ember's specialized datasets for mathematics (AIME), PhD-level science (GPQA), and competitive programming (Codeforces) evaluation.

## Quick Start

```python
from ember.api import datasets, DatasetBuilder, models

# Mathematics: AIME dataset
aime_data = datasets("aime")

# PhD-level science: GPQA dataset (requires HuggingFace authentication)
# First run: huggingface-cli login
try:
    gpqa_data = datasets("gpqa")
except Exception as e:
    print(f"Authentication error: {e}")

# Competitive programming: Codeforces dataset
cf_data = (
    DatasetBuilder()
    .from_registry("codeforces")
    .configure(difficulty_range=(800, 1200))
    .sample(5)
    .build()
)
```

## Dataset Details

### AIME (American Invitational Mathematics Examination)

Mathematical competition problems with clean numerical answers (0-999).

```python
from ember.api import datasets, models
from ember.core.utils.eval.numeric_answer import AIMEAnswerEvaluator

# Load dataset
aime_data = datasets("aime")

# Filter by contest
aime_i_data = (
    DatasetBuilder()
    .from_registry("aime")
    .configure(contest="I")  # I or II
    .build()
)

# Evaluate on single problem
problem = aime_data[0]
model = models.openai.gpt4o()
response = model(problem.query)

evaluator = AIMEAnswerEvaluator()
result = evaluator.evaluate(response, problem.metadata["correct_answer"])
print(f"Correct: {result.is_correct}")
```

### GPQA (Graduate-level Physics Questions and Answers)

PhD-level physics and chemistry questions with multistep reasoning.

```python
from ember.api import datasets, models
from ember.core.utils.eval.evaluators import MultipleChoiceEvaluator

# Authentication required - will raise GatedDatasetAuthenticationError if not authenticated
try:
    gpqa_data = datasets("gpqa")
except Exception as e:
    print(f"Run 'huggingface-cli login' first: {e}")
    # Request access at https://huggingface.co/datasets/Idavidrein/gpqa

# Filter by subject
physics_only = (
    DatasetBuilder()
    .from_registry("gpqa")
    .filter(lambda item: "physics" in item.metadata.get("subject", "").lower())
    .build()
)

# Evaluate
problem = gpqa_data[0] if gpqa_data else None
if problem:
    # Format prompt with choices
    prompt = problem.query + "\n\n"
    for key, choice in problem.choices.items():
        prompt += f"{key}. {choice}\n"
        
    model = models.openai.gpt4o()
    response = model(prompt)
    
    evaluator = MultipleChoiceEvaluator()
    result = evaluator.evaluate(response, problem.metadata["correct_answer"])
```

### Codeforces (Competitive Programming)

Algorithmic programming problems with test cases for evaluation.

```python
from ember.api import datasets, models
from ember.core.utils.eval.code_execution import CodeCompetitionEvaluator

# Load by difficulty (rating range)
cf_data = (
    DatasetBuilder()
    .from_registry("codeforces")
    .configure(difficulty_range=(800, 1200))  # Beginner-friendly
    .build()
)

# Example problem evaluation
problem = cf_data[0]
model = models.anthropic.claude_3_opus()
solution = model(f"Solve this programming problem and provide a Python solution:\n\n{problem.query}")

# Extract code from solution
import re
code_match = re.search(r"```python\n(.*?)```", solution, re.DOTALL)
if code_match:
    code = code_match.group(1)
    
    evaluator = CodeCompetitionEvaluator(language="python")
    result = evaluator.evaluate(code, problem.metadata["test_cases"])
    print(f"Tests passed: {result.is_correct}")
```

## Authentication for Gated Datasets

GPQA requires HuggingFace authentication:

1. **Authenticate** with `huggingface-cli login`
2. **Request access** at https://huggingface.co/datasets/Idavidrein/gpqa
3. **Handle errors** using try/except:

```python
from ember.core.exceptions import GatedDatasetAuthenticationError

try:
    gpqa_data = datasets("gpqa")
except GatedDatasetAuthenticationError as e:
    print(f"Authentication required: {e.recovery_hint}")
    # Will show: "Run `huggingface-cli login` to authenticate..."
except Exception as e:
    print(f"Other error: {e}")
```

## Batch Evaluation Example

Run multiple models on a dataset:

```python
from ember.api import datasets, models
from ember.core.utils.eval.numeric_answer import AIMEAnswerEvaluator

aime_data = datasets("aime")
problems = aime_data.sample(5)  # 5 random problems

model_configs = [
    ("gpt-4o", models.openai.gpt4o()),
    ("claude-3-opus", models.anthropic.claude_3_opus()),
]

evaluator = AIMEAnswerEvaluator()
results = {}

for name, model in model_configs:
    results[name] = []
    for problem in problems:
        response = model(problem.query)
        result = evaluator.evaluate(response, problem.metadata["correct_answer"])
        results[name].append(result.is_correct)
    
    accuracy = sum(results[name]) / len(results[name])
    print(f"{name}: {accuracy:.2%} accuracy")
```

## Performance Considerations

- **Cache dataset loading**: HuggingFace datasets are cached locally
- **Use sampling** for large datasets: `sample(n)` to work with subset
- **Preload for batch inference**: Fully load datasets before batch evaluation
- **Reuse evaluators**: Create evaluator instances once and reuse them

## Testing Your Setup

Run this to verify your environment is correctly set up:

```bash
uvx python -m ember.examples.data.new_datasets_example --skip-model-calls
```

The script checks dataset availability and provides helpful error messages if any issues are found.