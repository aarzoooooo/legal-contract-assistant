"""Evaluate retrieval and answer quality on a small hand-written question set.

Run from the project root:
    py eval/run_eval.py                    # retrieval + generated answers (calls Gemini, a few minutes)
    py eval/run_eval.py --retrieval-only   # retrieval only (fast, uses almost no quota)

Each question in eval/questions.json is either
  - answerable: has an "evidence" phrase copied from the contract (checked against the retrieved
    chunks) and an "answer_has" list of acceptable key facts (checked against the generated answer), or
  - not covered by the contract: the right behavior is to answer "I couldn't find this ...".
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MIN_SCORE, TOP_K            # noqa: E402
from src.generator import NOT_FOUND, answer_question  # noqa: E402
from src.loader import load_pdf                    # noqa: E402
from src.retriever import Retriever                # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def norm(text: str) -> str:
    """Lowercase, straighten curly quotes and collapse whitespace so matching is forgiving."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).lower()


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def is_refusal(answer: str) -> bool:
    """True if the answer opens with the standard "couldn't find this" sentence."""
    return norm(answer).startswith(norm(NOT_FOUND))


def ask(question: str, kept, retries: int = 3):
    """Generate an answer, waiting and retrying if the API is busy or rate limited."""
    last = None
    for attempt in range(retries):
        try:
            return answer_question(question, kept)["answer"].strip(), None
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 20 * (attempt + 1)
            print(f"      [retry] {type(e).__name__}; waiting {wait}s")
            time.sleep(wait)
    return None, str(last)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds to wait between Gemini calls")
    ap.add_argument("--k", type=int, default=TOP_K, help="how many chunks to retrieve")
    args = ap.parse_args()
    do_answers = not args.retrieval_only

    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))

    retrievers = {}
    for path in dict.fromkeys(q["contract"] for q in questions):
        print(f"Indexing {path} (first time can take a minute) ...")
        retrievers[path] = Retriever(load_pdf(ROOT / path))

    print(f"\nMIN_SCORE={MIN_SCORE}  k={args.k}  answers={'on' if do_answers else 'off'}\n")
    records = []

    for i, q in enumerate(questions, start=1):
        retriever = retrievers[q["contract"]]
        raw = retriever.retrieve(q["question"], k=args.k, min_score=0.0)
        kept = [(c, s) for c, s in raw if s >= MIN_SCORE]

        rec = {
            "n": i,
            "contract": Path(q["contract"]).name,
            "question": q["question"],
            "answerable": q["answerable"],
            "top_score": round(raw[0][1], 3),
            "top_section": raw[0][0].section,
            "kept": len(kept),
        }

        if q["answerable"]:
            ev = norm(q["evidence"])
            rec["rank"] = next((r for r, (c, _) in enumerate(raw, 1) if ev in norm(c.text)), None)
            rec["kept_hit"] = any(ev in norm(c.text) for c, _ in kept)
        else:
            rec["filtered"] = len(kept) == 0

        if do_answers:
            answer, error = ask(q["question"], kept) if kept else (NOT_FOUND, None)
            rec["answer"], rec["error"] = answer, error
            if answer is not None:
                if q["answerable"]:
                    low = norm(answer)
                    rec["answer_ok"] = not is_refusal(answer) and any(norm(x) in low for x in q["answer_has"])
                else:
                    rec["refused"] = is_refusal(answer)
            if kept:
                time.sleep(args.delay)

        records.append(rec)

        # one progress line per question
        if q["answerable"]:
            line = f"retrieval rank={rec['rank'] or '-'}"
            if do_answers:
                ok = rec.get("answer_ok")
                line += "  answer=" + ("PASS" if ok else "FAIL" if ok is False else "ERROR")
        else:
            how = "filtered by threshold" if rec["filtered"] else "passed threshold"
            line = f"not covered ({how}, top score {rec['top_score']})"
            if do_answers:
                ok = rec.get("refused")
                line += "  refused=" + ("PASS" if ok else "FAIL" if ok is False else "ERROR")
        print(f"[{i:2d}] {rec['contract'][:16]:<16} {line} | {q['question'][:58]}")

    # ------------------------------------------------------------------ summary
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]

    lines = [
        f"Questions: {len(records)} ({len(ans)} answerable, {len(unans)} not covered by the contract)",
        f"Settings: top-k={args.k}, relevance threshold={MIN_SCORE}",
        "",
        "Retrieval (answerable questions):",
        f"  evidence in top-1 chunk:             {pct(sum(r['rank'] == 1 for r in ans), len(ans))}",
        f"  evidence in top-{args.k} chunks:             "
        f"{pct(sum(r['rank'] is not None for r in ans), len(ans))}",
        f"  evidence kept after the threshold:   {pct(sum(r['kept_hit'] for r in ans), len(ans))}",
    ]
    if do_answers:
        graded_a = [r for r in ans if r.get("answer_ok") is not None]
        graded_u = [r for r in unans if r.get("refused") is not None]
        lines += [
            "",
            "Answers:",
            f"  answerable questions answered correctly:  {pct(sum(r['answer_ok'] for r in graded_a), len(graded_a))}",
            f"  uncovered questions correctly refused:    {pct(sum(r['refused'] for r in graded_u), len(graded_u))}",
            f"    stopped by the relevance threshold:     {sum(r['filtered'] for r in unans)}",
            f"    refused by the model:                   {sum(r['refused'] and not r['filtered'] for r in graded_u)}",
        ]
        errors = sum(1 for r in records if r.get("error"))
        if errors:
            lines.append(f"  (questions with API errors, not graded: {errors})")

    print("\n" + "=" * 64 + "\n" + "\n".join(lines) + "\n" + "=" * 64)

    failures = [
        r for r in records
        if (r["answerable"] and (r["rank"] is None or r.get("answer_ok") is False))
        or (not r["answerable"] and r.get("refused") is False)
    ]
    if failures:
        print("\nThings to look at:")
        for r in failures:
            print(f"  [{r['n']}] {r['question']}")
            print(f"       top match: {r['top_section'][:70]} ({r['top_score']})")
            if r.get("answer"):
                print(f"       answer: {r['answer'][:200].replace(chr(10), ' ')}")

    # ------------------------------------------------------------------ save results
    out = ROOT / "eval"
    (out / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Evaluation results", "", "```", *lines, "```", "",
          "| # | Contract | Question | Evidence rank | Answer |", "|---|---|---|---|---|"]
    for r in records:
        if r["answerable"]:
            rank = r["rank"] if r["rank"] else "missed"
            res = {True: "correct", False: "wrong", None: "n/a"}[r.get("answer_ok")] if do_answers else "-"
        else:
            rank = "not in contract"
            res = {True: "correctly refused", False: "answered anyway", None: "n/a"}[r.get("refused")] if do_answers else "-"
        md.append(f"| {r['n']} | {r['contract']} | {r['question']} | {rank} | {res} |")
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved {out / 'results.md'} and {out / 'results.json'}")


if __name__ == "__main__":
    main()
