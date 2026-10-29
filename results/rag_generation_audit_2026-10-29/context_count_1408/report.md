# Context-count RAG audit

## Methodology and controls

- DEV only: 56 questions; TEST was not used.
- Frozen Phase 1 rankings, dataset, prompt, adapter and deterministic generation were used.
- Retrieval top-k was 5; 1, 3 or 5 ranked contexts were requested and delivered as complete documents.
- Qwen2.5-1.5B-Instruct with the existing zero-shot LoRA adapter was loaded without retraining.
- One-context predictions were reused only when question, reference, context ID, prompt and configuration matched.
- Unique predictions: 336; reused: 0; new generations: 336.

- The 384-token predictions were not reused because the input budget changed; all 336 predictions were generated under the 1,408-token configuration.

## Aggregate results

```
retrieval_method  context_count  n_valid  EM_percent  F1_percent  generation_mean_seconds  input_truncated_percent  full_prompt_exceeds_budget_percent  context_truncated_percent  question_preserved_percent  mean_delivered_context_count
        semantic              1       56      71.429      78.465                    0.242                    0.000                               0.000                      0.000                     100.000                         1.000
        semantic              3       56      73.214      82.775                    0.276                    0.000                               0.000                      0.000                     100.000                         3.000
        semantic              5       56      76.786      83.416                    0.335                    0.000                               0.000                      0.000                     100.000                         5.000
          hybrid              1       56      71.429      77.770                    0.230                    0.000                               0.000                      0.000                     100.000                         1.000
          hybrid              3       56      69.643      79.076                    0.284                    0.000                               0.000                      0.000                     100.000                         3.000
          hybrid              5       56      67.857      79.603                    0.341                    0.000                               0.000                      0.000                     100.000                         5.000
```

The fixed input budget is enforced structurally: the instruction and question are kept complete, and contexts are included only as complete documents in ranking order. If the next complete document does not fit, it and all later documents are omitted. Thus input truncation no longer removes the question, and every delivered document is represented explicitly in the inclusion metadata.

All six configurations delivered every requested document completely in this DEV run, so the main comparisons are pure context-count comparisons under the fixed 1,408-token budget.

## Paired comparisons against one context

```
  method comparison  n_questions  mean_delta_F1  ci95_F1_low  ci95_F1_high  mean_delta_EM  ci95_EM_low  ci95_EM_high  f1_improved  f1_worsened  f1_unchanged  em_improved  em_worsened  em_unchanged
semantic     3_vs_1           56          0.043       -0.016         0.112          0.018       -0.054         0.089            5            4            47            3            2            51
semantic     5_vs_1           56          0.050       -0.035         0.141          0.054       -0.036         0.143            6            5            45            5            2            49
  hybrid     3_vs_1           56          0.013       -0.069         0.092         -0.018       -0.107         0.071            5            6            45            3            4            49
  hybrid     5_vs_1           56          0.018       -0.075         0.115         -0.036       -0.143         0.071            6            9            41            4            6            46
```

## Limitations and recommendation

- Retrieval latency is inherited from the frozen retrieval audit; it was not re-measured during generation.
- Results are DEV evidence for this model, prompt and context formatting, not TEST estimates.
- Semantic retrieval improved mean F1 by 0.043 with 3 contexts and 0.050 with 5 contexts, but both 95% bootstrap intervals include zero; this is positive DEV evidence, not statistically conclusive superiority.
- RRF increased F1 slightly with 3/5 contexts, but its EM deltas were negative and all F1 intervals include zero.
- More complete contexts increased generation latency; retrieval latency remains inherited from the frozen audit and was not re-measured.
