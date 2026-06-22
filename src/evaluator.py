evaluator1 = """You are evaluating the quality of a generated clinical note by another AI model, based on a real-world clinical conversation record.

Your task is to evaluate the generated note ONLY on the axis of:
STRICT CLINICAL GROUNDING AND NON-HALLUCINATORY DOCUMENTATION.

You must determine whether every clinical claim in the generated note is explicitly supported by the provided source material.

Criteria 1: EVALUATION OBJECTIVE

A high-quality output should:

- Contain only information EXPLICITLY SUPPORTED by the transcript, contextual notes, or clinician instructions.
- Faithfully represent the clinician's documented statements and observations.
- AVOID introducing unsupported medical conclusions or inferred diagnoses.
- Behave as a conservative clinical documentation assistant rather than an autonomous medical reasoner.

Criteria 2: ALLOWED BEHAVIOURS

The generated note MAY:

- Paraphrase conversational language into concise clinical language.
- Reorganize information into structured sections.
- Compress redundant information.
- Convert informal phrasing into EQUIVALENT clinical terminology.

Examples of acceptable transformations:

- "very tired" → "fatigue"
- "blood sugar problem" → "diabetes"
  ONLY if diabetes was explicitly mentioned.

Criteria 3: FAIL CONDITIONS

The generated note MUST be penalized if it contains:

- Unsupported diagnoses.
- Inferred diseases or conditions not explicitly stated in the source material.
- Fabricated examination findings.
- Invented vitals or laboratory results.
- Assumed normal findings.
- Unsupported causal explanations.
- Medically plausible completions not present in the source material.
- Added patient demographics or history not explicitly provided.

Examples of failures:

- "Iron deficiency anaemia"
  when only "low iron" was mentioned.

- "Normal physical examination"
  when no examination findings were documented.

- "Menopausal transition"
  when menopause was discussed speculatively but never diagnosed.

- "No other conditions identified"
  when this was never stated.

Criteria 4: SCORING

Return:
- A rationale: a brief explanation of your decision, citing specific examples from the model output and source material, that shows your reasoning.
- A binary score: 
    case 1: = 1 if the generated note is strictly grounded in the provided material.
    case 2: = 0 if the generated note contains any unsupported clinical inference, hallucination, fabrication, or medically inferred completion.

Be strict and conservative in your judgment.

Even medically reasonable inferences should FAIL if they are not explicitly supported by the source material.

Note: Starting with the following content is the MODEL INPUT. Remember to evaluate ONLY based on the provided source material, and do not infer beyond it. Read carefully and be conservative in your judgment.

--------------------------------------------------
{model_input}
--------------------------------------------------

Note: The model input ends here. Now, evaluate the model output based on the criteria above.

--------------------------------------------------
{model_output}
--------------------------------------------------

Note: The model output ends here. Now, provide your evaluation based on the criteria above.

Final Criteria: OUTPUT FORMAT

You must output ONLY valid JSON.

Use exactly this schema:

{
  "rationale": "short explanation of the decision",
  "score": 0
}

Do not output markdown.
Do not output code fences.
Do not output additional commentary."""

evaluator2 = """You are evaluating the quality of a generated clinical note.

Your task is to evaluate the generated note ONLY on the axis of:
CLINICAL INFORMATION EXTRACTION COMPLETENESS.

You must determine whether the generated note successfully captures the clinically relevant information explicitly discussed in the source material.

Criterion 1: EVALUATION OBJECTIVE

A high-quality output should:

- Capture the important clinically relevant details from the encounter.
- Preserve key symptoms, findings, history, investigations, treatments, and plans discussed during the consultation.
- Retain clinically meaningful contextual details that may affect future care.
- Produce a concise but sufficiently informative clinical summary.

This evaluator measures whether the generated note remains clinically useful after summarisation and compression.

Criterion 2: CORE REQUIREMENTS

The generated note should attempt to preserve:

- Chief complaints and major symptoms.
- Relevant symptom details such as severity, duration, progression, and associated symptoms.
- Relevant negatives explicitly discussed.
- Past medical history relevant to the encounter.
- Family history relevant to the encounter.
- Investigation results explicitly mentioned.
- Medication discussions and side effects.
- Treatment plans and clinician recommendations.
- Follow-up instructions or monitoring plans.
- Important contextual clinical reasoning explicitly discussed by the clinician.

Criterion 3: IMPORTANT DISTINCTIONS

This evaluator is NOT evaluating:

- Hallucinations or unsupported inference.
- Formatting quality or template adherence.
- Medical correctness of the clinician's decisions.
- Exhaustive verbatim transcription.

The generated note may paraphrase, reorganise, and compress information into concise clinical language.

Minor conversational filler or repeated dialogue does not need to be preserved.

This evaluator focuses on whether clinically important information was omitted.

Criterion 4: FAIL CONDITIONS

The generated note MUST be penalized if it:

- Omits major symptoms or complaints.
- Omits clinically relevant negatives.
- Omits important investigation results.
- Omits medication side effects or treatment discussions.
- Omits important family or medical history discussed during the encounter.
- Removes clinically important contextual information necessary for understanding the encounter.
- Over-compresses the encounter into a sparse or low-information summary.

--------------------------------------------------
EXAMPLES OF FAILURES
--------------------------------------------------

Examples of completeness failures include:

- Omitting fatigue despite repeated discussion of tiredness and low energy.

- Omitting the discussion that iron tablets previously caused constipation.

- Omitting that there were no clots during heavy bleeding.

- Omitting family history of diabetes.

- Omitting the monitoring discussion regarding recurrence of heavy periods.

- Omitting low iron investigation results.

Final criterion: SCORING

Return:
- A rationale: a brief explanation of your decision, citing specific examples from the model output and source material, that shows your reasoning.
- A binary score: 
    case 1: = 1 if the generated note is strictly grounded in the provided material.
    case 2: = 0 if the generated note contains any unsupported clinical inference, hallucination, fabrication, or medically inferred completion.

Be strict and conservative in your judgment.

Even medically reasonable inferences should FAIL if they are not explicitly supported by the source material.

Note: Starting with the following content is the MODEL INPUT. Remember to evaluate ONLY based on the provided source material, and do not infer beyond it. Read carefully and be conservative in your judgment.

--------------------------------------------------
<model_input>
{model_input}
</model_input>
--------------------------------------------------

Note: The model input ends here. Now, evaluate the model output based on the criteria above.

--------------------------------------------------
<model_output>
{model_output}
</model_output>
--------------------------------------------------

Note: The model output ends here. Now, provide your evaluation based on the criteria above.

Final Criteria: OUTPUT FORMAT

You must output ONLY valid JSON.

Use exactly this schema:

{
  "rationale": "short explanation of the decision",
  "score": 0
}

Do not output markdown.
Do not output code fences.
Do not output additional commentary."""

evaluator3 = """You are evaluating the quality of a generated clinical note.

Your task is to evaluate the generated note ONLY on the axis of:
CLINICAL INFORMATION EXTRACTION COMPLETENESS.

You must determine whether the generated note successfully captures the clinically relevant information explicitly discussed in the source material.

Criterion 1: EVALUATION OBJECTIVE

A high-quality output should:

- Capture the important clinically relevant details from the encounter.
- Preserve key symptoms, findings, history, investigations, treatments, and plans discussed during the consultation.
- Retain clinically meaningful contextual details that may affect future care.
- Produce a concise but sufficiently informative clinical summary.

This evaluator measures whether the generated note remains clinically useful after summarisation and compression.

Criterion 2: CORE REQUIREMENTS

The generated note should attempt to preserve:

- Chief complaints and major symptoms.
- Relevant symptom details such as severity, duration, progression, and associated symptoms.
- Relevant negatives explicitly discussed.
- Past medical history relevant to the encounter.
- Family history relevant to the encounter.
- Investigation results explicitly mentioned.
- Medication discussions and side effects.
- Treatment plans and clinician recommendations.
- Follow-up instructions or monitoring plans.
- Important contextual clinical reasoning explicitly discussed by the clinician.

Criterion 3: IMPORTANT DISTINCTIONS

This evaluator is NOT evaluating:

- Hallucinations or unsupported inference.
- Formatting quality or template adherence.
- Medical correctness of the clinician's decisions.
- Exhaustive verbatim transcription.

The generated note may paraphrase, reorganise, and compress information into concise clinical language.

Minor conversational filler or repeated dialogue does not need to be preserved.

This evaluator focuses on whether clinically important information was omitted.

Criterion 4: FAIL CONDITIONS

The generated note MUST be penalized if it:

- Omits major symptoms or complaints.
- Omits clinically relevant negatives.
- Omits important investigation results.
- Omits medication side effects or treatment discussions.
- Omits important family or medical history discussed during the encounter.
- Removes clinically important contextual information necessary for understanding the encounter.
- Over-compresses the encounter into a sparse or low-information summary.

--------------------------------------------------
EXAMPLES OF FAILURES
--------------------------------------------------

Examples of completeness failures include:

- Omitting fatigue despite repeated discussion of tiredness and low energy.

- Omitting the discussion that iron tablets previously caused constipation.

- Omitting that there were no clots during heavy bleeding.

- Omitting family history of diabetes.

- Omitting the monitoring discussion regarding recurrence of heavy periods.

- Omitting low iron investigation results.

Final criterion: SCORING

Return:
- A rationale: a brief explanation of your decision, citing specific examples from the model output and source material, that shows your reasoning.
- A binary score: 
    case 1: = 1 if the generated note is strictly grounded in the provided material.
    case 2: = 0 if the generated note contains any unsupported clinical inference, hallucination, fabrication, or medically inferred completion.

Be strict and conservative in your judgment.

Even medically reasonable inferences should FAIL if they are not explicitly supported by the source material.

Note: Starting with the following content is the MODEL INPUT. Remember to evaluate ONLY based on the provided source material, and do not infer beyond it. Read carefully and be conservative in your judgment.

--------------------------------------------------
<model_input>
{model_input}
</model_input>
--------------------------------------------------

Note: The model input ends here. Now, evaluate the model output based on the criteria above.

--------------------------------------------------
<model_output>
{model_output}
</model_output>
--------------------------------------------------

Note: The model output ends here. Now, provide your evaluation based on the criteria above.

Final Criteria: OUTPUT FORMAT

You must output ONLY valid JSON.

Use exactly this schema:

{
  "rationale": "short explanation of the decision",
  "score": 0
}

Do not output markdown.
Do not output code fences.
Do not output additional commentary."""

EVALUATORS = {
    "strict_grounding": evaluator1, 
    "information_completeness": evaluator2,
    "formatting_adherence": evaluator3
}

