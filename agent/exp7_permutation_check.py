# Experiment 7 (run after the first four code cells of vlm_table_agent.ipynb)
# 实验7：排列校验（每位评委的一列必须是 1..n 的排列）+ 修补 + 针对性重读
# 只用 Qwen，不调用 DeepSeek。在没调过的测试集（seed 123，12 张表 x mild/hard）上评测。
# 前提：notebook 前 4 个代码格已运行（模型、数据、工具）。方法在看结果前已固定。
import json, re
from collections import Counter

def n_songs_from(b):
    # 题目里写了"赋{n}到1分"，n 是去掉一首后剩下的歌曲数
    return int(re.search(r"赋(\d+)到1分", b["question"]).group(1)) + 1

def fix_columns(rows, n):
    """补齐短行(None)；某列恰好缺1个且其余是互不相同的1..n，就补上唯一缺的值；否则标记为坏列"""
    L = Counter(len(v) for v in rows.values()).most_common(1)[0][0]
    t = {k: (list(v[:L]) + [None] * (L - len(v))) for k, v in rows.items()}
    names = list(t); bad = []; repaired = 0
    for j in range(L):
        col = [t[k][j] for k in names]
        known = [x for x in col if x is not None]
        good = (all(isinstance(x, (int, float)) and float(x).is_integer() and 1 <= x <= n for x in known)
                and len(set(known)) == len(known))
        if not good:
            bad.append(j); continue
        missing = [i for i, x in enumerate(col) if x is None]
        if not missing:
            if sorted(known) != list(range(1, n + 1)): bad.append(j)
        elif len(missing) == 1 and len(known) == n - 1:
            t[names[missing[0]]][j] = (set(range(1, n + 1)) - set(known)).pop(); repaired += 1
        else:
            bad.append(j)
    return t, bad, repaired

def validate(out, b):
    n = n_songs_from(b)
    if "table" not in out or not out["table"]: return None
    try:
        t, bad, rep = fix_columns(out["table"], n)
    except Exception:
        return None
    names_ok = len(t) == n and b["removed"] in t and (b["target"] is None or b["target"] in t)
    return {"table": t, "bad": bad, "repaired": rep, "names_ok": names_ok, "valid": names_ok and not bad}

def pipeline(b, path, max_reads=3):
    n = n_songs_from(b); hint, reads = "", []
    for k in range(max_reads):
        out = read_table(path, hint)
        v = validate(out, b)
        reads.append({"hint": hint, "raw_table": out.get("table"), "v": v})
        if v and v["valid"]: break
        parts = [f"表格共{n}行，每行对应一首歌（行名是歌曲名，不是评委），每行的数字是该歌曲从各评委得到的分数。"]
        if v and v["names_ok"] and v["bad"]:
            parts.append("第" + "、".join(str(j + 1) for j in v["bad"]) + "位评委的分数有误，请重新核对这几列。")
        hint = "".join(parts)
    return reads

def final_of(reads, b):
    """从多次读取里选：行名对且坏列最少的（平手取最早的）。返回 (答案, 表, 是否通过校验)"""
    cands = [r for r in reads if r["v"]]
    if not cands: return None, None, False
    best = min(cands, key=lambda r: (not r["v"]["names_ok"], len(r["v"]["bad"])))
    v = best["v"]
    return (solve(b, v["table"]) if v["names_ok"] else None), v["table"], v["valid"]

fresh = make_bench(123, 12, "f")
exp7 = []
for lvl in ["mild", "hard"]:
    for i, b in enumerate(fresh):
        src = b["image"].replace(".png", f"_{lvl}.jpg")
        degrade(b["image"], src, lvl)
        reads = pipeline(b, src)
        raw0 = reads[0]["raw_table"]
        A_ans = solve(b, raw0) if raw0 else None
        v0 = reads[0]["v"]
        P0_ans = solve(b, v0["table"]) if (v0 and v0["names_ok"]) else None
        P_ans, P_table, P_valid = final_of(reads, b)
        row = {"level": lvl, "i": i, "truth": b["answer"],
               "A_ok": is_ok(A_ans, b["answer"]), "A_exact": raw0 == b["true_table"],
               "P0_ok": is_ok(P0_ans, b["answer"]), "P0_exact": bool(v0) and v0["table"] == b["true_table"],
               "P_ok": is_ok(P_ans, b["answer"]), "P_exact": P_table == b["true_table"],
               "P_valid": P_valid, "n_reads": len(reads), "repaired_first": (v0 or {}).get("repaired", 0),
               "reads": reads}
        exp7.append(row)
        print(f"[{lvl}] f{i+1}: truth={row['truth']} | A {'OK' if row['A_ok'] else 'X'}(exact={row['A_exact']}) "
              f"| P0 {'OK' if row['P0_ok'] else 'X'}(exact={row['P0_exact']}, repaired={row['repaired_first']}) "
              f"| P {'OK' if row['P_ok'] else 'X'}(exact={row['P_exact']}, valid={P_valid}, reads={row['n_reads']})")

print()
for lvl in ["mild", "hard"]:
    rs = [r for r in exp7 if r["level"] == lvl]
    s = lambda k: sum(r[k] for r in rs)
    print(f"{lvl} (n={len(rs)}) | table exact A {s('A_exact')}, P0 {s('P0_exact')}, P {s('P_exact')} "
          f"| answer correct A {s('A_ok')}, P0 {s('P0_ok')}, P {s('P_ok')} "
          f"| P passes check on {s('P_valid')} | mean reads {sum(r['n_reads'] for r in rs)/len(rs):.2f}")
print()
for lvl in ["mild", "hard"]:
    for flag in [True, False]:
        rs = [r for r in exp7 if r["level"] == lvl and r["P_valid"] == flag]
        if rs:
            print(lvl, "P passes check" if flag else "P fails check", len(rs), "images -> P table exact", sum(r["P_exact"] for r in rs),
                  "| answer correct", sum(r["P_ok"] for r in rs))
print("自检（应与 README 的 Exp5 一致）：A table exact mild 11 / hard 7；A answer correct mild 11 / hard 9")
json.dump(exp7, open("/content/exp7_permutation.json", "w"), ensure_ascii=False, indent=2)
