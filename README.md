# Trustworthy-Medical-AI-Notetaker

Medical note generation systems are increasingly used to convert patient-provider conversations into structured clinical documentation. However, these systems introduce safety risks including hallucinated medical information, omission of critical patient details, and formatting inconsistencies that may impact downstream clinical workflows.

This project presents an evaluation framework for assessing the reliability and safety of AI medical note generators. The framework focuses on three key dimensions:

- **Grounding and Hallucination Detection** — ensuring generated notes contain only information supported by the original conversation.
- **Information Extraction Completeness** — measuring whether clinically relevant information is preserved.
- **Documentation Compliance** — verifying adherence to required note structures and formatting standards.
- You may also add your own dimensions if you think they are important! Customize it in `src/evaluator.py`

We will choose a set of generator models to produce medical notes from clicinal conversation inputs, and then use closed-weights LLM-based judges to evaluate the generated notes across the above dimensions. The evaluation results will be analyzed to identify the most reliable **generator** and **judge models**, and to understand the strengths and weaknesses of different approaches.

The framework supports automated evaluation using multiple LLM-based judges and can be used to benchmark medical note generation systems across diverse clinical conversations.

## How to run

1. - Configure your API keys in '.env' file
    - install dependencies with `pip install -r requirements.txt`.
    - Python 3.10+ is required.
    - Nvidia GPU is needed for fine-tuning or running generators locally, but not required. The evaluation framework can run with all API-based models as generators and judges, without needing to run any model locally.

2. Specifiy your desired generators, judges and evaluation criterias.

```python
from src.evaluator import EVALUATORS

generators = {
    "Qwen-2.5": "qwen-2.5",
    "DS-R1": "ds-r1",
    ...
}

judges = {
    "GPT-4o": "gpt-4o",
    "Claude-3.5": "claude-3.5",
    ...
}
```

3. Prepare your input data ready.

    - The input should include:
        - a strict system prompt restricting the output guidelines
        - a user prompt containing the actual medical conversations to be summarized
    - Then, choose your preferred generator models to generate the responses given the input. The output should be the generated medical summarizing notes.
    - Then call the judge models to evaluate the qualities of the generated notes based on the evaluation criteria.
    - Summarize the judge scores into a 4D array with dimensions [num_judges, num_inputs, num_generators, num_evaluation_criteria].
    - Refer to [documentation](documentation/instructions.ipynb) for an example workflow.

For more detailed documentations of how the evaluation framework works, please refer to the [documentation](documentation/instructions.ipynb).

(Note: actual example data will be provided soon, stay tuned!)

4. Finally, use the `JudgeGeneratorScorer` to analyze the scores and determine the best judge and generator. `scores` is the 4d array from last step.

```python
from scorer import JudgeGeneratorScorer

scorer = JudgeGeneratorScorer(
    scores,                             # your 4D array
    judge_names  = judges.keys(), # e.g., ["GPT-4o", ...]
    gen_names    = generators.keys(), # e.g., ["Qwen-2.5", ...]
    eval_names   = EVALUATORS.keys(), # e.g., ["Hallucination", "Adherence", "Coherence"]
    eval_weights = [2.0, 1.0, 1.0],    # optional: prioritize eval dims. In a mdedical context, hallucination is believed to be the most important, so weigthed twice here
)

result = scorer.select()               # runs everything

# The answers you care about:
result.best_judge                      # 'Claude-3.5'
result.best_generator                  # 'Qwen-2.5'       (highest mean)
result.best_generator_balanced         # 'DS-R1'          (strongest weakest link)
result.best_generator_per_evaluator    # {'Hallucination': 'Qwen-2.5', ...}

# Full tables if you want to dig in:
result.judge_ranking                   # pd.DataFrame with all 4 signals
result.generator_ranking               # pd.DataFrame with per-eval scores
result.judge_weights                   # {'Claude-3.5': 0.502, 'Gemini-Flash': 0.0, ...}

result.summary()                       # compact human-readable printout
```

5. After determining the best generator and judge, you can use their outputs to train a smaller instruction-following model with RLHF.

I fine-tuned a Qwen3-4B-Instruct model with LoRA on the generated data with the best generator and judge. The training results and analysis will be released soon.

```bash
python src/train.py 
    --train_data your_training_data.jsonl 
    --output_dir ./fine_tuned_model 
    --model_id Qwen/Qwen3-4B-Instruct-2507 
    --val_split 0.1
```

This framework can also extend to a broader range of medical AI evaluation scenarios, such as medical question answering, diagnosis prediction, and treatment recommendation. By systematically evaluating the reliability and safety of medical AI systems, we can identify their limitations and guide future improvements to ensure they provide trustworthy support in clinical practice.

## Future steps

- We will soon release a set of example data and evaluation results. Stay tuned!
- In addition to just quantitative analysis, we will soon enable qualitative analysis of the evaluation results, to provide more detailed insights into why the judges prefer certain generators over others, and to identify specific strengths and weaknesses of different approaches. This can inform future improvements in model design and training.

## License

MIT License. See [LICENSE](LICENSE) for details.
