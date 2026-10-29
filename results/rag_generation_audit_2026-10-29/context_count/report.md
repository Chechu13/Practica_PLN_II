# Context-count RAG audit

## Methodology and controls

- DEV only: 56 questions; TEST was not used.
- Frozen Phase 1 rankings, dataset, prompt, adapter and deterministic generation were used.
- Retrieval top-k was 5; only the first 1, 3 or 5 ranked contexts were delivered.
- Qwen2.5-1.5B-Instruct with the existing zero-shot LoRA adapter was loaded without retraining.
- One-context predictions were reused only when question, reference, context ID, prompt and configuration matched.
- Unique predictions: 336; reused: 112; new generations: 224.

## Aggregate results

```
retrieval_method  context_count  n_valid  EM_percent  F1_percent  generation_mean_seconds  input_truncated_percent
        semantic              1       56      71.429      76.947                    0.271                    0.000
        semantic              3       56       5.357       8.381                    1.004                   92.857
        semantic              5       56       0.000       1.238                    1.095                  100.000
          hybrid              1       56      69.643      74.467                    0.288                    0.000
          hybrid              3       56       3.571       4.265                    0.944                   94.643
          hybrid              5       56       0.000       1.034                    0.995                  100.000
```

The 3- and 5-context configurations show a large quality decline because the fixed `max_length=384` input limit truncates 92.9% or more of inputs; the question can therefore be removed from the model input. This is a prompt-length confound, not evidence that additional evidence is intrinsically harmful.

## Paired comparisons against one context

```
  method comparison  n_questions  mean_delta_F1  ci95_F1_low  ci95_F1_high  mean_delta_EM  ci95_EM_low  ci95_EM_high  f1_improved  f1_worsened  f1_unchanged  em_improved  em_worsened  em_unchanged
semantic     3_vs_1           56         -0.686       -0.796        -0.571         -0.661       -0.786        -0.536            0           44            12            0           37            19
semantic     5_vs_1           56         -0.757       -0.855        -0.652         -0.714       -0.821        -0.589            0           48             8            0           40            16
  hybrid     3_vs_1           56         -0.702       -0.808        -0.591         -0.661       -0.786        -0.536            0           45            11            0           37            19
  hybrid     5_vs_1           56         -0.734       -0.835        -0.628         -0.696       -0.821        -0.571            0           47             9            0           39            17
```

## Limitations and recommendation

- Retrieval latency is inherited from the frozen retrieval audit; it was not re-measured during generation.
- Results are DEV evidence for this model, prompt and context formatting, not TEST estimates.
- Before testing multiple contexts as a quality intervention, increase the input budget or apply a pre-validated context compression policy while keeping the question visible.
