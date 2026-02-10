from __future__ import annotations

from pathlib import Path

from ingestion import build_corpus
from search import search
from answering import build_answer_v2


def main() -> None:
    data_dir = Path("data")
    corpus = build_corpus(data_dir)

    print(f"Indexed {len(corpus)} chunks.")
    counts = {}
    for c in corpus:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = search(corpus, query, top_k=40)
        answer, citations = build_answer_v2(query, results)

        print("\n" + "=" * 80)
        print(f"Question: {query}\n")
        print("Answer:")
        print(answer)
        print("\nCitations:")
        for c in citations:
            print(f"- {c}")
        print("=" * 80)


if __name__ == "__main__":
    main()
