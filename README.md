# vlm-eval-notes

Evaluating Qwen2.5-VL-3B on table reading and multi-step reasoning: pure VLM (0/8) vs VLM + Python tool (8/8)

## Setup
- Model: Qwen/Qwen2.5-VL-3B-Instruct (Hugging Face Transformers), 4-bit NF4 (bitsandbytes)
- Hardware: Google Colab free T4 GPU (15 GB)
- Task: "remove one song, re-rank the rest per judge, then sum the scores"
- Test set: 8 synthetic tables (matplotlib images) with randomized sizes
  (4-5 songs, 5-7 judges) and scores; ground truth computed by code, greedy decoding
- Notebook: `qwen25vl_eval.ipynb`, raw outputs: `results.json`
- Section 3 of the notebook is a separate qualitative demo on a lecture-slide
  screenshot (`screenshot1.png`, not included because it is course material):
  reading numbers and grounding a bounding box. It is not part of the 8-question
  score and is skipped automatically if the file is missing.

## Results

| Method | Correct |
|---|---|
| Pure VLM (reads image and reasons, chain-of-thought prompt) | 0 / 8 |
| VLM reads table to JSON, Python does the ranking and summing | 8 / 8 |

Failure analysis of the pure-VLM answers (see `pure_out` in `results.json`):

| Failure type | Questions | Evidence |
|---|---|---|
| Impossible answer (exceeds the maximum achievable total) | Q2, Q5, Q6, Q8 | 171 (max 20), 27 (max 24), 191 (max 28), 51 (max 21) |
| Skipped re-ranking, reported raw totals from the original table | Q1, Q3, Q7 | Q1 = A's original total (16, which also exceeds the max of 15); Q3 = A/E's original total; Q7 = the removed song B's own total |
| Invented an inconsistent re-ranking | Q4 | 12 matches no original total and cannot be derived from the table |

The maximum achievable total is (points for the top rank) x (number of judges),
e.g. Q2: 4 songs remain, 5 judges, so at most 4 x 5 = 20.

## Engineering notes
1. **float32 did not fit on a T4.** The 3B model needs about 13 GB in float32;
   `device_map="auto"` offloaded layers to CPU, which made generation far
   slower. With 4-bit NF4 quantization the model fits in about 2.4 GB and a
   typical answer takes a few seconds.
2. **Grounding coordinates live in the resized image's space.** Boxes must be
   rescaled by (original size / resized size) before drawing on the original.
3. **Config matters.** The same question can give different answers under
   different precision settings (float32 with CPU offload vs 4-bit), so all
   reported numbers use 4-bit NF4 only.
4. **Answer parsing matters.** An early parser missed outputs like
   "最终答案是17" ("the final answer is 17", written in Chinese by the model)
   and scored them as failures. The parser now matches the last number after
   "最终答案", and all outputs were re-scored.

## Limitations
- Only 8 questions of a single task type; conclusions do not generalize to
  "reasoning" in general.
- The 8/8 comes from a hand-written solver plus VLM extraction. It shows the
  VLM reads these tables reliably, not that tool use makes the model smarter.
- Only one 3B model at one quantization level was tested; a larger model might
  close the gap without tools.
- Synthetic tables are clean and rendered by matplotlib; real slides or
  photos would be harder.

## Follow-up: agent experiments

[`agent/`](agent/) turns the fixed "VLM reads table, Python computes" pipeline into an agent loop (DeepSeek as the
brain, Qwen2.5-VL as the table reader, a Python solver as a tool) and stress-tests it on degraded images.
Main finding: on clean tables the agent reaches 8/8, but on degraded images it is not clearly better than the
fixed workflow; the bottleneck is visual reading, not planning. Details, negative results and limitations are in
[`agent/README.md`](agent/README.md).

Next: make the agent abstain when views disagree, and test on real screenshots.
