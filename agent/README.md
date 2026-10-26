# Table-reading VLM agent

Follow-up to the [evaluation in the repo root](../README.md), which found that Qwen2.5-VL-3B reads score tables reliably
but fails the multi-step arithmetic (pure VLM 0/8; VLM reads the table and Python computes 8/8).
Here the fixed pipeline is replaced by an **agent loop**, and then stress-tested on degraded images to see what actually helps.

- Notebook: [`vlm_table_agent.ipynb`](vlm_table_agent.ipynb) (Colab, T4 GPU)
- Raw per-question results: `results/*.json`

## Architecture

```
question + image path
        |
   DeepSeek (brain, API)  <----------------------------+
        | decides which tool to call                   | tool result (table, warnings, totals)
        v                                              |
  read_table(image, hint) -- Qwen2.5-VL-3B, 4-bit, local GPU
  compute_totals(table, removed) -- deterministic Python solver, also returns the maximum possible total
        |
   model decides when to stop -> "final answer: N"
```

- The **agent** is the loop in `run_agent`: the model chooses the next tool from the previous result and decides when to answer.
  A fixed workflow (`run_fixed`) reads once and computes, with no checking or retry.
- Task: *remove one song, re-rank the rest per judge, sum the scores* on synthetic matplotlib tables
  (4-5 songs, 5-7 judges). Ground truth is computed by code.
- Degraded images: `mild` (downscale, light blur, JPEG) and `hard` (more downscaling, blur, small rotation, noise).

## Results

All numbers are single runs. The DeepSeek calls are sampled, so individual outcomes can change between runs; Qwen is greedy and deterministic.

### Clean images (8 tables)

| Method | Correct |
|---|---|
| Pure VLM, no tools (root README) | 0 / 8 |
| Agent loop (DeepSeek + Qwen reader + solver) | 8 / 8 |

Agent cost: about 0.0009 USD per question (about 2,700 input and 770 output tokens on average; 0.0069 USD for all 8 at deepseek-flash off-peak prices).

### Degraded images, same 8 tables

| Method | mild | hard | total |
|---|---|---|---|
| Fixed workflow: read once, compute | 6/8 | 6/8 | 12/16 |
| Agent v1 (checks shape, may re-read) | 7/8 | 6/8 | 13/16 |
| Scripted retry with checksum (v2) | 5/8 | 6/8 | 11/16 |
| Agent v2 with checksum | 5/8 | 7/8 | 12/16 |

Cost: agent v1 0.022 USD, agent v2 0.052 USD for 16 questions each (more re-reads cost more).

### Reader diagnostic: is the first read of the table exactly right? (8 images per level)

| | without Tot. | without Tot., 2x upscale | with Tot. | with Tot., 2x upscale |
|---|---|---|---|---|
| mild | 5 | 8 | 4 | 2 |
| hard | 6 | 5 | 4 | 0 |

### Fresh test set: multi-view consistency (12 new tables x 2 levels, methods fixed before running)

A = read the original; B = read a 2x upscale; C = read original, 2x and 3x, use a table at least two views agree on.

| | table exactly right (A / B / C) | answer correct (A / B / C) | views agree |
|---|---|---|---|
| mild | 11 / 11 / 11 | 11 / 11 / 11 | 12 of 12 |
| hard | 7 / 7 / 9 | 9 / 8 / 10 | 9 of 12 |

| Views agree? | Images | C table exactly right |
|---|---|---|
| yes | 21 | 19 |
| no | 3 | 1 |

## What I found

1. **Replacing the fixed pipeline with an agent loop works on clean images** (8/8), at under 0.001 USD per question.
2. **On degraded images the agent is not clearly better than the fixed workflow** (13/16 vs 12/16, one question). The first read is identical in both
   (Qwen is deterministic), so the agent can only win through a re-read that changes the table. The one clear example is a mild image where the
   fixed workflow got no usable table and the agent recovered after 3 reads; in other cases re-reading did not fix the table.
3. **A checksum did not help.** Reading the `Tot.` column made the reader worse (first-read accuracy dropped), and re-reading with a
   text hint rarely fixed a wrong table. The scripted retry and the agent behaved alike (11/16 vs 12/16), so I cannot attribute any gain to the
   agent's own decisions. Caveat: the code assumes `Tot.` is the last number in each row, so part of the drop may come from that parsing assumption.
4. **The bottleneck looks like perception, not planning.** Hints and checks add no new information for the reader; changing what the reader sees can.
5. **Not everything replicated.** On the first set, a 2x upscale raised mild first-read accuracy from 5/8 to 8/8. On the fresh set it made no
   difference (11/12 either way), so that effect was probably specific to a few hard tables.
6. **Agreement between views is a suggestive uncertainty signal.** When two of three views agreed, the table was exactly right in 19 of 21 cases;
   when none agreed, in 1 of 3. That flags 2 of the 4 wrong tables on the fresh set, with 1 false alarm. Only 3 images were in the "no agreement" group,
   so this is a lead to test, not an established result.
7. **Answer accuracy overstates reading quality.** Several answers were correct even though the table was misread (for example 9 correct answers
   but only 7 exactly-right tables on the hard fresh set).
8. One agent re-read seen on a clean image in an early run did not reproduce when the same question was re-run, so it is not counted as evidence of self-correction.

## Limitations

- Small samples (8 tables x 2 levels; 12 tables x 2 levels), single runs; differences of one or two questions are within noise.
- Synthetic, cleanly drawn tables with artificial degradation, not real screenshots or photos.
- One reader model (Qwen2.5-VL-3B, 4-bit) and one brain model (DeepSeek flash).
- The first four experiments reuse the same 8 tables, so methods were partly tuned while looking at them; only the last experiment is a held-out set.
- The agent's loop is simple (two tools, no memory, no planning beyond tool choice).

## Next steps

- Make the agent **abstain or ask for a better image** when views disagree, instead of answering, and measure coverage vs accuracy.
- Give the agent tools that provide new information to the reader (crop and zoom on a row, a second reader model) rather than text hints.
- Test on real screenshots, and repeat each agent run several times to measure variance.

## Result files

| File in `results/` | Experiment | Name when re-running the notebook |
|---|---|---|
| `agent_results.json` | Agent on clean images | `exp1_clean.json` |
| `robust_results.json` | Degraded images: fixed workflow vs agent v1 | `exp2_degraded.json` |
| `v2_results.json` | Checksum: scripted retry vs agent v2 | `exp3_checksum.json` |
| `diag_results.json` | Reader diagnostic (Tot. column, 2x upscale) | `exp4_diagnostic.json` |
| `fresh_results.json` | Multi-view consistency on a fresh test set | `exp5_fresh.json` |

The files in `results/` come from the original runs. Their fields are slightly different from the re-runnable notebook
(for example, the first agent run did not record the fixed-workflow baseline on clean images).

## Reproduce

1. Open the notebook in Colab with a **T4 GPU** runtime.
2. Add a Colab secret `DEEPSEEK_API_KEY` and enable notebook access (never commit a key).
3. Run all cells. The API cost of the whole notebook is roughly 0.08 USD (the sum of the per-experiment costs above); Qwen runs locally.
