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

The whole notebook was run twice (a second time from a fresh Colab session). Qwen is greedy and gave identical outputs both times, so every Qwen-only
result (fixed workflow, reader diagnostic, multi-view experiment) reproduced exactly. DeepSeek calls are sampled, so agent results moved by 0-1 questions between runs;
both runs are reported where they differ.

### Clean images (8 tables)

| Method | Correct |
|---|---|
| Pure VLM, no tools (root README) | 0 / 8 |
| Fixed workflow: read once, compute | 8 / 8 (both runs) |
| Agent loop (DeepSeek + Qwen reader + solver) | 8 / 8 (run 1), 7 / 8 (run 2) |

In run 2 the agent got Q7 wrong (11) while the fixed workflow got it right (14). I first suspected DeepSeek mis-copied the table when passing it to
`compute_totals`, and tested this: with the true tables as the reader output (no Qwen), the agent passed the table to the solver correctly in 24 of 24 runs
(8 tables x 3 runs) and answered all 24 correctly. So copying errors are not a likely cause (at most a small rate, with only 24 trials).
The cause of the Q7 miss is unexplained; one remaining candidate is that the agent re-read the image with a hint that changed Qwen's output, but I did not save that run's call log.

Agent cost: about 0.0009 USD per question (about 2,700 input and 770 output tokens on average; 0.0069 USD for all 8 at deepseek-flash off-peak prices).

### Degraded images, same 8 tables

Agent rows show run 1 / run 2; the other rows were identical in both runs.

| Method | mild | hard | total |
|---|---|---|---|
| Fixed workflow: read once, compute | 6/8 | 6/8 | 12/16 |
| Agent v1 (checks shape, may re-read) | 7/8 / 8/8 | 6/8 / 6/8 | 13/16 / 14/16 |
| Scripted retry with checksum (v2) | 5/8 | 6/8 | 11/16 |
| Agent v2 with checksum | 5/8 / 6/8 | 7/8 / 6/8 | 12/16 / 12/16 |

Cost: agent v1 about 0.022 USD, agent v2 0.04-0.05 USD for 16 questions each (more re-reads cost more).

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

### Permutation check on the fresh set (Qwen only, no LLM; 12 new tables x 2 levels)

Each judge gives every song a distinct score, so every judge's column must be a permutation of 1..n. The check validates each read against this,
fills in a single missing value when it is uniquely determined, and otherwise re-reads (up to 3 reads) with a hint that states the number of song rows and
which judges' columns are inconsistent. Methods were fixed before running; the set is the held-out one used above (same A as in the multi-view experiment).
A = read once; P0 = check and repair the first read only; P = check, repair and targeted re-read.

| | table exactly right (A / P0 / P) | answer correct (A / P0 / P) | P passes the check |
|---|---|---|---|
| mild | 11 / 11 / 12 | 11 / 11 / 12 | 12 of 12 |
| hard | 7 / 7 / 7 | 9 / 9 / 9 | 8 of 12 |

As an error flag on the first read (24 images): 6 first reads were wrong, the check flagged 5 of them with no false alarms; 18 of the 19 reads that passed were exactly right.
If the pipeline abstains whenever the check fails, it answers 20 of 24 questions and gets 19 right; answering everything gives 21/24, and reading once gives 20/24.

### Scripted-retry control (Qwen only, no LLM)

Can the agent's one clear gain (mild Q8) be reproduced by plain code? In mild Q4 and Q8 the first read parses as valid JSON, but the row keys are judge labels
(`J1`..`J7`) instead of song names (in Q8 the first row is even repeated 7 times), so the solver rejects the table and the fixed workflow returns no answer.

| Control | mild | hard |
|---|---|---|
| Fixed workflow (read once) | 6/8 | 6/8 |
| Retry when JSON fails to parse or row lengths differ (up to 3 reads) | 6/8, never triggered | 6/8, never triggered |
| Retry when the solver fails, generic hint (up to 3 reads) | Q4, Q8 still no answer after 3 reads | - |
| Same, hint states the number of song rows (taken from the question) | Q4: structure fixed, one digit wrong (answer 14, truth 13); Q8: structure fixed, last value of row D missing, solver still fails | - |

Nothing in the rule-based controls reproduced the agent's correct Q8 answer (14). The hints were written after looking at these two failures, so this is a lower bound on what rules can do, not a fair ceiling.

## What I found

1. **Replacing the fixed pipeline with an agent loop works on clean images** (8/8 in run 1, 7/8 in run 2), at under 0.001 USD per question,
   but it did not beat the fixed workflow (8/8 both runs). One miss in 8 is within run-to-run noise (the agent is sampled), and I could not trace its cause.
2. **On degraded images the agent is not clearly better than the fixed workflow** (13/16 and 14/16 vs 12/16, a one-to-two question gap that
   changed between runs). The first read is identical in both (Qwen is deterministic), so the agent can only win through a re-read that changes the table.
   The clear example is mild Q8: the first read had judge labels as row names, so the fixed and scripted versions returned no answer, and the agent
   recovered and answered 14 after several reads. A rule-based retry with a shape hint fixed the structure but left the table incomplete (one value missing) and did not
   reproduce this, so I cannot say how the agent got there: the agent's call log was not saved. One possibility, not verified, is that it filled in the missing value
   by reasoning (each judge's scores are a permutation of 1..n). In other cases re-reading did not fix the table (mild Q4 stayed wrong, and Q2 was wrong for every method).
   This is one question.
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
9. **Reproducibility.** Re-running the full notebook in a new session reproduced every Qwen-only number exactly; only the DeepSeek-driven results moved.

10. **A free consistency check flagged most misreads, but re-reading rarely fixed them.** On the fresh set the permutation check flagged 5 of the 6 wrong first reads
    with no false alarms. Looking at the logged reads: the one miss (hard f2) had its last judge column dropped (4 judges instead of 5), and every remaining column was a valid permutation,
    so a per-column check cannot see it. The one image fixed by re-reading (mild f5) was a transposed read (judge labels as row names), fixed by the hint stating the number of song rows, not by a
    column-specific hint. On the 4 flagged hard images, re-reading did not help: in 3 of them Qwen returned exactly the same table despite the new hint, and in the fourth the re-read dropped a column.
    So on hard images the check works as an "I am not sure" signal rather than a fix. The single-missing-value repair never triggered on this set, so it has only been tested on simulated reads.
    This is 24 images, one run (Qwen is deterministic), and the check exploits this task's structure (distinct scores per judge); real tables may not offer such a constraint.

## Limitations

- Small samples (8 tables x 2 levels; 12 tables x 2 levels), single runs; differences of one or two questions are within noise.
- Synthetic, cleanly drawn tables with artificial degradation, not real screenshots or photos.
- One reader model (Qwen2.5-VL-3B, 4-bit) and one brain model (DeepSeek flash).
- The first four experiments reuse the same 8 tables, so methods were partly tuned while looking at them; only experiments 5 and 7 use a held-out set (seed 123).
- The agent's loop is simple (two tools, no memory, no planning beyond tool choice).
- The brain (DeepSeek) is a text-only model that never sees the image; it only sees Qwen's text output. This is a modular design with the VLM as a perception tool, not an end-to-end multimodal agent.
- The rule-based retry controls used hints written after seeing the failures on these tables, and were run only on the questions where the first read failed.
- The permutation check depends on this task's structure, was evaluated on 24 images with one missed error, and its repair step was never exercised on a real read.

## Next steps

- Save the full tool-call log of every agent run (reads, hints, solver inputs) so that a single miss like clean-image Q7 can be traced afterwards; copying tables to the solver was ruled out as a common cause (24/24 correct).
- Run the permutation check on the original 8 tables to compare it with the agent on the same questions, and feed its warnings to the agent as a tool result.
- Test the same idea on tables without a built-in constraint (for example row totals or units) to see how far it generalises.
- Make the agent **abstain or ask for a better image** when views disagree, instead of answering, and measure coverage vs accuracy.
- Give the agent tools that provide new information to the reader (crop and zoom on a row, a second reader model) rather than text hints.
- Test on real screenshots, and repeat each agent run several times to measure variance.

## Code and raw results

- Experiments 1-5: [`vlm_table_agent.ipynb`](vlm_table_agent.ipynb).
- Experiment 6 (scripted-retry control): [`exp6_scripted_retry_control.py`](exp6_scripted_retry_control.py), run after the first four code cells of the notebook.
- Experiment 7 (permutation check): [`exp7_permutation_check.py`](exp7_permutation_check.py), run the same way.
- Raw per-question results (including every read and hint for experiments 6-7): `results/`.

## Reproduce

1. Open the notebook in Colab with a **T4 GPU** runtime.
2. Add a Colab secret `DEEPSEEK_API_KEY` and enable notebook access (never commit a key).
3. Run all cells. The API cost of the whole notebook is roughly 0.08 USD (the sum of the per-experiment costs above); Qwen runs locally.
