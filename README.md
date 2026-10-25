# vlm-eval-notes

Evaluating Qwen2.5-VL-3B on table reading and multi-step reasoning: pure VLM (0/8) vs VLM + Python tool (8/8)

## Setup
- Model: Qwen/Qwen2.5-VL-3B-Instruct (Hugging Face Transformers), 4-bit NF4 (bitsandbytes)
- Hardware: Google Colab free T4 GPU (15 GB)
- Task: "remove one song, re-rank the rest per judge, then sum the scores"
- Test set: 8 questions (1 lecture slide + 7 synthetic tables with randomized
  sizes and scores; ground truth computed by code), greedy decoding
- Notebook: `qwen25vl_eval.ipynb`, raw outputs: `results.json`

## Results

| Method | Correct |
|---|---|
| Pure VLM (reads image and reasons, chain-of-thought prompt) | 0 / 8 |
| VLM reads table to JSON, Python does the ranking and summing | 8 / 8 |

| Failure type | Questions | Evidence |
|---|---|---|
| Impossible answer (exceeds the maximum achievable total) | Q2, Q5, Q6, Q8 | 171 (max 20), 27 (max 24), 191 (max 28), 51 (max 21) |
| Skipped re-ranking, reported raw totals from the original table | Q1, Q3, Q7 | Q1 = A's original total; Q3 = A/E's original total; Q7 = the removed song B's own total |
| Invented an inconsistent re-ranking | Q4 | 12 matches no original total and cannot be derived from the table |

## Engineering notes
1. **float32 did not fit on a T4.** The 3B model needs about 13 GB in float32;
   `device_map="auto"` offloaded layers to CPU, making generation take 1-2 min.
   With 4-bit NF4 quantization the model fits in 2.4 GB and a similar
   generation takes about 4 s.
2. **Grounding coordinates live in the resized image's space.** Boxes must be
   rescaled by (original size / resized size) before drawing on the original.
3. **Config matters.** The same question gave different answers under
   float32+offload and 4-bit (21 vs 15), so all reported numbers use 4-bit.
4. **My first answer parser missed outputs like "The final answer is 17"** (the model wrote it in Chinese as `最终答案是17`) and scored
   them as failures. Fixed by loosening the regex and re-scoring all outputs.
   
## Limitations
- Only 8 questions of a single task type; conclusions do not generalize to
  "reasoning" in general.
- The 8/8 comes from a hand-written solver plus VLM extraction. It shows the
  VLM reads these tables reliably, not that tool use makes the model smarter.
- Only one 3B model at one quantization level was tested; a larger model might
  close the gap without tools.
- Synthetic tables are clean and rendered by matplotlib; real slides or
  photos would be harder.

Next: compare against a larger VLM, and test noisier table images.
