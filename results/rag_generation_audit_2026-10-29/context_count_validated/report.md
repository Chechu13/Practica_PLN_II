# Context-count RAG audit

## Methodology and controls

- DEV only: 56 questions; TEST was not used.
- Frozen Phase 1 rankings, dataset, prompt, adapter and deterministic generation were used.
- Retrieval top-k was 5; 1, 3 or 5 ranked contexts were requested, then the fixed-budget policy determined how many were fully or partially delivered.
- Qwen2.5-1.5B-Instruct with the existing zero-shot LoRA adapter was loaded without retraining.
- One-context predictions were reused only when question, reference, context ID, prompt and configuration matched.
- Unique predictions: 336; reused: 107; new generations: 229.

- Five one-context rows from the previous audit exceeded the validated budget and were regenerated; therefore the one-context metrics here are the corrected validated baseline, not a byte-for-byte reuse of every previous row.

## Aggregate results

```
retrieval_method  context_count  n_valid  EM_percent  F1_percent  generation_mean_seconds  input_truncated_percent  full_prompt_exceeds_budget_percent  context_truncated_percent  question_preserved_percent  mean_delivered_context_count
        semantic              1       56      73.214      78.733                    0.271                    0.000                               3.571                      3.571                     100.000                         1.000
        semantic              3       56      62.500      76.230                    0.256                    0.000                              92.857                     92.857                     100.000                         2.500
        semantic              5       56      62.500      75.635                    0.250                    0.000                             100.000                    100.000                     100.000                         2.571
          hybrid              1       56      73.214      78.038                    0.268                    0.000                               5.357                      5.357                     100.000                         1.000
          hybrid              3       56      66.071      74.906                    0.247                    0.000                              94.643                     94.643                     100.000                         2.429
          hybrid              5       56      66.071      74.906                    0.248                    0.000                             100.000                    100.000                     100.000                         2.482
```

The fixed input budget is enforced structurally: the instruction and question are kept complete, and only the ranked context prefix is partially included or omitted when necessary. Thus input truncation no longer removes the question. The number of delivered contexts can be below the requested count when the budget is exhausted; this is recorded per row and is part of the treatment definition.

Because the fixed budget permits only a ranked prefix, the 3- and 5-requested conditions are not equivalent to delivering three or five complete documents: their mean delivered counts are reported above. A future experiment that requires all five documents must use a larger input budget or a separate compression policy.

## Paired comparisons against one context

```
  method comparison  n_questions  mean_delta_F1  ci95_F1_low  ci95_F1_high  mean_delta_EM  ci95_EM_low  ci95_EM_high  f1_improved  f1_worsened  f1_unchanged  em_improved  em_worsened  em_unchanged
semantic     3_vs_1           56         -0.025       -0.086         0.031         -0.107       -0.214        -0.018            3            8            45            1            7            48
semantic     5_vs_1           56         -0.031       -0.091         0.026         -0.107       -0.196        -0.018            3            9            44            1            7            48
  hybrid     3_vs_1           56         -0.031       -0.115         0.051         -0.071       -0.179         0.036            4            8            44            3            7            46
  hybrid     5_vs_1           56         -0.031       -0.115         0.051         -0.071       -0.179         0.036            4            8            44            3            7            46
```

## Limitations and recommendation

- Retrieval latency is inherited from the frozen retrieval audit; it was not re-measured during generation.
- Results are DEV evidence for this model, prompt and context formatting, not TEST estimates.
- Before testing multiple contexts as a quality intervention, increase the input budget or apply a pre-validated context compression policy while keeping the question visible.
