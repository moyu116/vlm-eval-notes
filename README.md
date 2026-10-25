# vlm-eval-notes

Evaluating Qwen2.5-VL-3B on table-reading and multi-step reasoning, and testing
whether offloading computation to code fixes its failures.

## Setup
- Model: Qwen/Qwen2.5-VL-3B-Instruct (Hugging Face Transformers), 4-bit NF4 (bitsandbytes)
- Hardware: Google Colab free T4 GPU (15 GB)
- Task: "remove one song, re-rank the rest per judge, then sum the scores"
- Test set: 9 questions (1 lecture slide + 8 synthetic tables with randomized
  sizes and scores; ground truth computed by code), greedy decoding
- Notebook: `qwen25vl_eval.ipynb`, raw outputs: `results.json`

## Results

| Method | Correct |
|---|---|
| Pure VLM (reads image and reasons, chain-of-thought prompt) | 0 / 9 |
| VLM reads table to JSON, Python does the ranking and summing | 9 / 9 |

## Failure analysis (pure VLM)
| Pattern | Questions |
|---|---|
| Answer exceeds the mathematical upper bound (invented formulas / numbers) | 5 / 9 |
| Skips re-ranking and reports an original total instead | 2 / 9 |
| Computes the wrong song | 1 / 9 |
| Arithmetic and re-scoring becomes inconsistent mid-way | 1 / 9 |

In every pure-VLM trace the model could copy the table correctly but did not
actually carry out the re-ranking step.

## Engineering notes
1. **float32 did not fit on a T4.** The 3B model needs about 13 GB in float32;
   `device_map="auto"` offloaded layers to CPU, making generation take 1-2 min.
   With 4-bit NF4 quantization the model fits in 2.4 GB and a similar
   generation takes about 4 s.
2. **Grounding coordinates live in the resized image's space.** Boxes must be
   rescaled by (original size / resized size) before drawing on the original.
3. **Config matters.** The same question gave different answers under
   float32+offload and 4-bit (21 vs 15), so all reported numbers use 4-bit.
4. **My first answer parser missed "最终答案是17"-style outputs** and scored
   them as failures. Fixed by loosening the regex and re-scoring all outputs.

## Limitations
- Only 9 questions of a single task type; conclusions do not generalize to
  "reasoning" in general.
- The 9/9 comes from a hand-written solver plus VLM extraction. It shows the
  VLM reads these tables reliably, not that tool use makes the model smarter.
- Only one 3B model at one quantization level was tested; a larger model might
  close the gap without tools.
- Synthetic tables are clean and rendered by matplotlib; real slides or
  photos would be harder.

Next: compare against a larger VLM, and test noisier table images.
