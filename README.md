# vlm-eval-notes

Evaluating Qwen2.5-VL-3B on lecture-slide screenshots, and testing whether
offloading computation to code fixes its reasoning failures.

## Setup
- Model: Qwen/Qwen2.5-VL-3B-Instruct (Hugging Face Transformers)
- Hardware: Google Colab free T4 GPU (15GB)
- Notebook: `qwen25vl_eval.ipynb`

## Findings

| Test | Result |
|---|---|
| Read numbers from a table | Accurate, but incomplete (missed some cells) |
| Read table + answer a question whose answer is printed on the slide | Correct |
| Multi-step reasoning (re-rank and re-score after removing a candidate) | Wrong: copied original totals instead of re-ranking |
| Same question with step-by-step prompting | Cleaner output, still wrong: same judge gave equal scores to several songs; totals contradicted earlier steps |
| Text grounding (bounding box of a title) | Right region, imprecise box (covered ~60% of the text width) |
| **VLM reads table → Python computes** | **Correct** (A=13, C=18, D=11) |

## Key takeaways
1. **Precision matters for speed.** `bfloat16` on a T4 hung for 8+ minutes
   (no native support, software emulation); `float32` finished in ~12 s.
2. **Reading ≠ reasoning.** The 3B model reads tables reliably but fails at
   multi-step symbolic reasoning, and prompt engineering did not fix it.
3. **Tool use helps.** Letting the model only extract data and delegating
   ranking/summing to Python turned a wrong answer into a correct one.
4. **Grounding coordinates live in the resized image's space.** Boxes must be
   rescaled by (original size / resized size) before drawing on the original.

## Limitations
Only one task was tested so far (n=1), so no success rates are claimed.
Next: expand to 10+ slides and compare pure-VLM vs. VLM+tools accuracy.
