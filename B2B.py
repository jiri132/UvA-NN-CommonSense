import math
import csv
from collections import Counter
import pandas as pd

# ---- Stopwords ----
STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "i", "my",
             "it", "its", "to", "of", "and", "or", "not", "can", "do",
             "in", "on", "at", "for", "with", "this", "that", "be"}

# ---- Load CSVs ----
def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def load_data(data_path, answers_path):
    data    = {row["id"]: row for row in load_csv(data_path)}
    answers = {row["id"]: row["answer"] for row in load_csv(answers_path)}
    
    combined = []
    for id_, row in data.items():
        combined.append({
            "id":         id_,
            "false_sent": row["FalseSent"],
            "A":          row["OptionA"],
            "B":          row["OptionB"],
            "C":          row["OptionC"],
            "answer":     answers.get(id_, None)
        })
    return combined

def stem(word):
    suffixes = ["ing", "tion", "ness", "ment", "ful", "less", "ly", "es", "ed", "er", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

# ---- Preprocessing ----
def tokenize(text, use_stem=False):
    tokens = [
        w.lower().strip(".,!?\"'")
        for w in text.split()
        if w.lower().strip(".,!?\"'") not in STOPWORDS
    ]
    return [stem(w) for w in tokens] if use_stem else tokens

# ---- Build IDF from full corpus ----
def build_idf(all_sentences):
    N = len(all_sentences)
    df = Counter()
    for sent in all_sentences:
        for word in set(tokenize(sent, use_stem=False)):
            df[word] += 1
    idf = {word: math.log(N / (1 + count)) for word, count in df.items()}
    return idf

# ---- Cosine similarity ----
def cosine_sim(tokens1, tokens2):
    c1, c2 = Counter(tokens1), Counter(tokens2)
    vocab   = set(c1) | set(c2)
    dot     = sum(c1[w] * c2[w] for w in vocab)
    mag1    = math.sqrt(sum(v**2 for v in c1.values()))
    mag2    = math.sqrt(sum(v**2 for v in c2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def score_pair(false_sent, candidate, idf, stem_false=False, stem_cand=False):
    false_tokens = tokenize(false_sent, stem_false)
    weighted_false = []
    for word in false_tokens:
        weight = max(1, int(idf.get(word, 0) * 10))
        weighted_false.extend([word] * weight)
    candidate_tokens = tokenize(candidate, stem_cand)
    return cosine_sim(weighted_false, candidate_tokens)

# ---- Predict best option ----
def predict(row, idf):
    labels = ["A", "B", "C"]

    combos = {
        "RAW->RAW":   (False, False),
        "RAW->STEM":  (False, True),
        "STEM->RAW":  (True,  False),
        "STEM->STEM": (True,  True),
    }

    all_scores = {}
    all_winners = {}
    all_margins = {}

    for name, (sf, sc) in combos.items():
        scores = {
            label: score_pair(row["false_sent"], row[label], idf, sf, sc)
            for label in labels
        }
        all_scores[name] = scores
        sorted_vals = sorted(scores.values(), reverse=True)
        winner = max(scores, key=scores.get)
        all_winners[name] = winner
        all_margins[name] = sorted_vals[0] - sorted_vals[1]  # gap over 2nd

    # step 1 — if all 4 agree, easy win
    if len(set(all_winners.values())) == 1:
        return all_winners["RAW->RAW"], all_scores, "all_agreed"

    # step 2 — pick the combo that was MOST confident (biggest margin)
    best_combo = max(all_margins, key=all_margins.get)
    return all_winners[best_combo], all_scores, f"{best_combo}_won"
# ---- Evaluate accuracy ----
def evaluate(dataset, idf):
    correct = 0
    results = []
    for row in dataset:
        predicted, all_scores, method = predict(row, idf)
        is_correct = predicted == row["answer"]
        if is_correct:
            correct += 1
        results.append({
            "id":        row["id"],
            "false":     row["false_sent"],
            "A":         row["A"],
            "B":         row["B"],
            "C":         row["C"],
            "predicted": predicted,
            "answer":    row["answer"],
            "correct":   is_correct,
            "method":    method,
            "scores":    all_scores,   # ← replaces scores_raw / scores_stem
        })
    accuracy = correct / len(dataset) * 100
    return results, accuracy


# ---- Main ----
if __name__ == "__main__":
    print("Loading data...")
    dataset = load_data("./data/train_data.csv", "./data/train_answers.csv")

    print("Building IDF from corpus...")

    # use ALL sentences (false + all candidates) for IDF
    all_sentences = []
    for row in dataset:
        all_sentences.extend([row["false_sent"], row["A"], row["B"], row["C"]])
    idf = build_idf(all_sentences)

    df = pd.DataFrame(idf,[0])
    df = df.T.reset_index()
    df.columns = ["Word", "IDF-output"]
    # df.to_excel("idf_output_stemmed.xlsx", index=False)
    print(df)    

    print("Running predictions...\n")
    results, accuracy = evaluate(dataset, idf)

    for r in results[:10]:
        status = "✅" if r["correct"] else "❌"
        print(f"{status} [{r['id']}]")
        print(f"   False:     {r['false']}")
        print(f"   Tokens:    {tokenize(r['false'])} (raw)")
        print(f"              {tokenize(r['false'], use_stem=True)} (stem)")
        print(f"   Predicted: {r['predicted']}  |  Correct: {r['answer']}  |  Method: {r['method']}")
        for combo, scores in r['scores'].items():
            winner = max(scores, key=scores.get)
            print(f"   --- {combo} ---")
            for label in ["A", "B", "C"]:
                marker = "←" if label == winner else ""
                print(f"      {label}: {scores[label]:.4f}  {tokenize(r[label], 'STEM' in combo.split('->')[1])}  {marker}")
        print()
    print(f"Accuracy: {accuracy:.2f}% over {len(dataset)} samples")