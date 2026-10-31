# Experiment 6 (scripted-retry control; run after the first four code cells of vlm_table_agent.ipynb)
# 对照实验（只用 Qwen，不调用 DeepSeek，API 费用为 0）：
# 固定流程 + "读不出表/行长不一致就用代码写好的 hint 重读"，不用 LLM。
# 问题：Agent 在读不出表时的那点收益，不用 LLM 能不能拿到？
# 前提：已经运行过 notebook 前 4 个代码格（模型、数据 bench、工具函数 read_table/solve/degrade）

from collections import Counter

def scripted_retry(b, image_path, max_reads=3):
    log, table_final, hint = [], None, ""
    for k in range(max_reads):
        out = read_table(image_path, hint)
        log.append({"hint": hint, "ok": "table" in out, "warnings": out.get("warnings", []),
                    "table": out.get("table"), "raw": out.get("raw")})
        if "table" in out:
            table_final = out["table"]
            if not out["warnings"]:
                break                      # 读到了且行长一致：结束
            # 行长不一致：用众数行长写 hint（只用图里读出的信息，不用真实答案）
            n = Counter(out["row_lengths"].values()).most_common(1)[0][0]
            hint = f"共{len(out['table'])}行，每行{n}个分数。"
        else:
            hint = "输出必须是合法JSON，例如 {\"A\": [1, 2, 3]}。"   # 解析失败：提示格式
    return solve(b, table_final) if table_final else None, table_final, log

ctrl = []
for lvl in ["mild", "hard"]:
    for i, b in enumerate(bench):
        dst = f"/content/eval/g{i+1}_{lvl}.jpg"
        degrade(b["image"], dst, lvl)
        fx_ans, fx_out = run_fixed(b, dst)
        sc_ans, sc_table, log = scripted_retry(b, dst)
        ctrl.append({"level": lvl, "i": i, "truth": b["answer"],
                     "fixed": fx_ans, "fixed_ok": is_ok(fx_ans, b["answer"]),
                     "retry": sc_ans, "retry_ok": is_ok(sc_ans, b["answer"]),
                     "retry_table_exact": sc_table == b["true_table"], "n_reads": len(log), "log": log})
        r = ctrl[-1]
        print(f"[{lvl}] Q{i+1}: truth={r['truth']} | fixed={r['fixed']} {'OK' if r['fixed_ok'] else 'X'}"
              f" | retry={r['retry']} {'OK' if r['retry_ok'] else 'X'} (exact={r['retry_table_exact']}, reads={r['n_reads']})")
print()
for lvl in ["mild", "hard"]:
    rs = [r for r in ctrl if r["level"] == lvl]
    print(f"{lvl}: fixed {sum(r['fixed_ok'] for r in rs)}/8 | scripted retry {sum(r['retry_ok'] for r in rs)}/8 "
          f"(table exact {sum(r['retry_table_exact'] for r in rs)}/8, re-reads on {sum(r['n_reads'] > 1 for r in rs)} questions)")
print("参考（README）：agent v1 mild 7/8 或 8/8，hard 6/8；fixed 6/8，6/8")
json.dump(ctrl, open("/content/exp6_retry_control.json", "w"), ensure_ascii=False, indent=2)


# ---- 6b: retry only when the solver fails, generic hint (run on the questions that had no answer) ----
def retry_on_solver_error(b, image_path, max_reads=3):
    hint, log, ans = "", [], None
    for k in range(max_reads):
        out = read_table(image_path, hint)
        ans = solve(b, out["table"]) if "table" in out else None
        log.append({"hint": hint, "table": out.get("table"), "raw": out.get("raw"), "answer": ans})
        if ans is not None:
            break
        if "table" in out:
            need = [b["removed"]] + ([b["target"]] if b["target"] else [])
            hint = f"上一次读到的行名是{list(out['table'])}，但题目涉及歌曲{need}，请逐行核对行标签和数字。"
        else:
            hint = "输出必须是合法JSON，例如 {\"A\": [1, 2, 3]}。"
    return ans, log

ctrl2 = []
for r in ctrl:
    if r["retry"] is None:
        b = bench[r["i"]]
        ans, log = retry_on_solver_error(b, f"/content/eval/g{r['i']+1}_{r['level']}.jpg")
        ctrl2.append({"level": r["level"], "i": r["i"], "truth": r["truth"], "answer": ans,
                      "ok": is_ok(ans, r["truth"]), "n_reads": len(log), "log": log})
        print(f"[{r['level']}] Q{r['i']+1}: truth={r['truth']} | retry-on-error={ans} {'OK' if is_ok(ans, r['truth']) else 'X'} (reads={len(log)})")
json.dump(ctrl2, open("/content/exp6b_retry_on_error.json", "w"), ensure_ascii=False, indent=2)

# ---- 6c: same, but the hint states the number of song rows (taken from the question) ----
def retry_with_shape(b, image_path, max_reads=3):
    n_songs = int(re.search(r"赋(\d+)到1分", b["question"]).group(1)) + 1
    hint, log, ans = "", [], None
    for k in range(max_reads):
        out = read_table(image_path, hint)
        ans = solve(b, out["table"]) if "table" in out else None
        log.append({"hint": hint, "table": out.get("table"), "answer": ans})
        print(f"   read {k+1}: hint={hint!r}\n     table={out.get('table')} -> answer={ans}")
        if ans is not None:
            break
        hint = (f"表格共{n_songs}行，每行对应一首歌（行名是歌曲名，不是评委），"
                f"每行的数字是该歌曲从各评委得到的分数。")
    return ans, log

ctrl3 = []
for r in ctrl2:
    b = bench[r["i"]]
    print(f"[{r['level']}] Q{r['i']+1} truth={r['truth']} true_table={b['true_table']}")
    ans, log = retry_with_shape(b, f"/content/eval/g{r['i']+1}_{r['level']}.jpg")
    ctrl3.append({"level": r["level"], "i": r["i"], "truth": r["truth"], "answer": ans,
                  "ok": is_ok(ans, r["truth"]), "n_reads": len(log), "log": log})
    print(f"   => {ans} {'OK' if is_ok(ans, r['truth']) else 'X'}\n")
json.dump(ctrl3, open("/content/exp6c_retry_with_shape.json", "w"), ensure_ascii=False, indent=2)
