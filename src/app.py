# src/app.py
from .qa_engine import answer

if __name__ == "__main__":
    while True:
        q = input("Question: ")
        if not q:
            break
        res = answer(q)
        print(res.text)
        print("\nCitations:", res.citations)
