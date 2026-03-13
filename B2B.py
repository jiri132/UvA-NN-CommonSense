import csv
import math
import random
import numpy as np
from collections import Counter

# =============================================
#  SEEDS — fixed for reproducibility
# =============================================
random.seed(42)
np.random.seed(42)

# =============================================
#  HYPERPARAMETERS
# =============================================
LEARNING_RATE = 0.025
EPOCHS        = 8       # early stopping — test peaked around epoch 6
HIDDEN_DIM    = 32
MARGIN        = 0.3
MAX_VOCAB     = 600
LAMBDA_REG    = 0.001   # L2 weight decay

# =============================================
#  FILE PATHS — change these if needed
# =============================================
TRAIN_DATA_PATH    = "./data/train_data.csv"
TRAIN_ANSWERS_PATH = "./data/train_answers.csv"
TEST_DATA_PATH     = "./data/test_data.csv"
OUTPUT_PATH        = "./data/predictions.csv"

# =============================================
#  STOPWORDS
# =============================================
STOPWORDS = {
    "a","an","the","is","are","was","were","i","my","it","its","to","of",
    "and","or","not","can","do","in","on","at","for","with","this","that",
    "be","have","has","had","will","would","could","should","may","might"
}

# =============================================
#  TOKENIZER
# =============================================
def tokenize(text):
    return [
        w.lower().strip(".,!?\"'")
        for w in text.split()
        if w.lower().strip(".,!?\"'") not in STOPWORDS
        and w.lower().strip(".,!?\"'") != ""
    ]

# =============================================
#  LOAD TRAINING DATA (has answers)
# =============================================
def load_train(data_path, answers_path):
    data, answers = {}, {}
    with open(data_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            data[row["id"]] = row
    with open(answers_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            answers[row["id"]] = row["answer"]
    return [
        {
            "id":         id_,
            "false_sent": row["FalseSent"],
            "A":          row["OptionA"],
            "B":          row["OptionB"],
            "C":          row["OptionC"],
            "answer":     answers[id_]
        }
        for id_, row in data.items()
    ]

# =============================================
#  LOAD TEST DATA (no answers)
# =============================================
def load_test(data_path):
    with open(data_path, newline='', encoding='utf-8') as f:
        return [
            {
                "id":         row["id"],
                "false_sent": row["FalseSent"],
                "A":          row["OptionA"],
                "B":          row["OptionB"],
                "C":          row["OptionC"],
            }
            for row in csv.DictReader(f)
        ]

# =============================================
#  BUILD VOCAB + IDF
#
#  Built from ALL sentences — train + test
#  candidates. Safe because candidates are
#  text only, not labels.
#
#  Keeps middle IDF range:
#  - skip words in >80% of sentences (too common)
#  - skip words in <5  sentences     (too rare)
# =============================================
def build_vocab_idf(all_rows):
    all_sentences = []
    for row in all_rows:
        all_sentences.append(tokenize(row["false_sent"]))
        for l in ["A", "B", "C"]:
            all_sentences.append(tokenize(row[l]))

    counts  = Counter(w for s in all_sentences for w in s)
    N       = len(all_sentences)
    df      = Counter(w for s in all_sentences for w in set(s))
    idf     = {w: math.log(N / (1 + df[w])) for w in counts if counts[w] >= 3}

    idf_min  = math.log(N / (N * 0.8))
    idf_max  = math.log(N / 5)
    filtered = {w: v for w, v in idf.items() if idf_min < v < idf_max}

    # sort by frequency — common meaningful words first
    by_count = sorted(filtered, key=lambda w: counts[w], reverse=True)[:MAX_VOCAB]
    vocab    = {w: i for i, w in enumerate(by_count)}
    return vocab, idf

# =============================================
#  TF-IDF WEIGHTED ONE-HOT SEQUENCE
#
#  Each word → (vocab_index, tfidf_weight)
#  One non-zero per RNN step — no sparsity
# =============================================
def to_sequence(text, vocab, idf):
    tokens = tokenize(text)
    tf     = Counter(tokens)
    total  = max(len(tokens), 1)
    return [
        (vocab[w], (tf[w] / total) * idf.get(w, 0))
        for w in tokens
        if w in vocab and (tf[w] / total) * idf.get(w, 0) > 0
    ]

# =============================================
#  MAKE BINARY TRAINING PAIRS
#
#  Each row → 3 pairs (one per candidate)
#  correct candidate → label 1
#  wrong candidates  → label 0
# =============================================
def make_binary(train_data, vocab, idf):
    pairs = []
    for row in train_data:
        correct = row["answer"]
        for l in ["A", "B", "C"]:
            pairs.append({
                "seq_s": to_sequence(row["false_sent"], vocab, idf),
                "seq_e": to_sequence(row[l],            vocab, idf),
                "label": 1 if l == correct else 0
            })
    return pairs

def make_triplets(train_data, vocab, idf):
    # First, collect all possible answers to sample negatives from
    all_possible_answers = []
    for row in train_data:
        for l in ["A", "B", "C"]:
            all_possible_answers.append(to_sequence(row[l], vocab, idf))
            
    triplets = []
    for row in train_data:
        # q = FalseSent, pos = the ground truth option
        seq_q = to_sequence(row["false_sent"], vocab, idf)
        correct_label = row["answer"] # e.g., 'A'
        seq_pos = to_sequence(row[correct_label], vocab, idf)
        
        # Sample a random negative answer from the entire answer space
        # (Excluding empty sequences if any)
        seq_neg = random.choice(all_possible_answers)
        while len(seq_neg) == 0: 
            seq_neg = random.choice(all_possible_answers)
            
        triplets.append({
            "q": seq_q,
            "pos": seq_pos,
            "neg": seq_neg
        })
    return triplets

# =============================================
#  RNN — INITIALISE
#
#  Wx: [V x H] — word → hidden weights
#  Wh: [H x H] — hidden → hidden weights
#  b:  [H]     — bias
# =============================================
def init_rnn(V):
    Wx = np.random.randn(V, HIDDEN_DIM) * math.sqrt(1.0 / V)
    Wh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * math.sqrt(1.0 / HIDDEN_DIM)
    b  = np.zeros(HIDDEN_DIM)
    return Wx, Wh, b

# =============================================
#  RNN FORWARD — INFERENCE
#  Fast, no stored states
# =============================================
def rnn_encode(seq, Wx, Wh, b):
    h = np.zeros(HIDDEN_DIM)
    for idx, w in seq:
        h = np.tanh(Wh @ h + Wx[idx] * w + b)
    return h

# =============================================
#  RNN FORWARD — TRAINING
#  Stores all hidden states needed for BPTT
# =============================================
def rnn_forward_full(seq, Wx, Wh, b):
    if not seq:
        return np.zeros(HIDDEN_DIM), [], [], []
    h, hs, zs = np.zeros(HIDDEN_DIM), [], []
    for idx, w in seq:
        z = Wh @ h + Wx[idx] * w + b
        h = np.tanh(z)
        hs.append(h.copy())
        zs.append(z.copy())
    return h, hs, zs, seq

# =============================================
#  COSINE SIMILARITY
# =============================================
def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

# =============================================
#  BACKPROP THROUGH TIME (BPTT)
# =============================================
def bptt(dh, hs, zs, xs, Wx, Wh):
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db  = np.zeros(HIDDEN_DIM)
    for t in reversed(range(len(hs))):
        dz       = dh * (1.0 - hs[t] ** 2)
        db      += dz
        idx, w   = xs[t]
        dWx[idx] += dz * w
        h_prev   = hs[t - 1] if t > 0 else np.zeros(HIDDEN_DIM)
        dWh     += np.outer(dz, h_prev)
        dh       = np.clip(Wh.T @ dz, -1.0, 1.0)
    return dWx, dWh, db

# =============================================
#  CONTRASTIVE LOSS + WEIGHT UPDATE
#
#  label=1: loss = (1 - sim)^2     push sim → 1
#  label=0: loss = max(0, sim - margin)^2
#                                   push sim below margin
#  L2 regularisation on Wx and Wh
# =============================================
def train_step(seq_f, seq_c, label, Wx, Wh, b):
    h_f, hs_f, zs_f, xs_f = rnn_forward_full(seq_f, Wx, Wh, b)
    h_c, hs_c, zs_c, xs_c = rnn_forward_full(seq_c, Wx, Wh, b)

    if not hs_f or not hs_c:
        return 0.0

    sim = cosine(h_f, h_c)

    if label == 1:
        loss  = (1.0 - sim) ** 2
        d_sim = -2.0 * (1.0 - sim)
    else:
        v = sim - MARGIN
        if v <= 0:
            return 0.0
        loss  = v ** 2
        d_sim = 2.0 * v

    nf, nc = np.linalg.norm(h_f), np.linalg.norm(h_c)
    if nf < 1e-10 or nc < 1e-10:
        return loss

    dn, dot = nf * nc, np.dot(h_f, h_c)
    dh_f = d_sim * (h_c / dn - h_f * dot / (nf ** 2 * dn))
    dh_c = d_sim * (h_f / dn - h_c * dot / (nc ** 2 * dn))

    dWx_f, dWh_f, db_f = bptt(dh_f, hs_f, zs_f, xs_f, Wx, Wh)
    dWx_c, dWh_c, db_c = bptt(dh_c, hs_c, zs_c, xs_c, Wx, Wh)

    Wx -= LEARNING_RATE * (np.clip(dWx_f + dWx_c, -1.0, 1.0) + LAMBDA_REG * Wx)
    Wh -= LEARNING_RATE * (np.clip(dWh_f + dWh_c, -1.0, 1.0) + LAMBDA_REG * Wh)
    b  -= LEARNING_RATE * np.clip(db_f + db_c, -1.0, 1.0)

    return loss

def train_step_triplet(seq_q, seq_pos, seq_neg, Wx, Wh, b):
    # Forward pass for all three components
    h_q, hs_q, zs_q, xs_q = rnn_forward_full(seq_q, Wx, Wh, b)
    h_pos, hs_pos, zs_pos, xs_pos = rnn_forward_full(seq_pos, Wx, Wh, b)
    h_neg, hs_neg, zs_neg, xs_neg = rnn_forward_full(seq_neg, Wx, Wh, b)

    # Calculate Cosine Similarities
    sim_pos = cosine(h_q, h_pos)
    sim_neg = cosine(h_q, h_neg)

    # Triplet Loss: max(0, M - sim_pos + sim_neg)
    val = MARGIN - sim_pos + sim_neg
    
    if val <= 0:
        # Margin is satisfied, no gradient update needed
        return 0.0
    
    loss = val
    
    # Gradients of the loss with respect to similarities
    # dL/d_sim_pos = -1, dL/d_sim_neg = 1
    d_sim_pos = -1.0
    d_sim_neg = 1.0

    # Helper to get dh from d_sim (Chain rule for cosine)
    def get_dh(h1, h2, d_sim):
        n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
        if n1 < 1e-10 or n2 < 1e-10: return np.zeros_like(h1)
        dn = n1 * n2
        dot = np.dot(h1, h2)
        return d_sim * (h2 / dn - h1 * dot / (n1**2 * dn))

    # Calculate gradients for hidden states
    # Note: h_q receives gradients from both the positive and negative comparisons
    dh_q_pos = get_dh(h_q, h_pos, d_sim_pos)
    dh_q_neg = get_dh(h_q, h_neg, d_sim_neg)
    dh_q = dh_q_pos + dh_q_neg
    
    dh_pos = get_dh(h_pos, h_q, d_sim_pos)
    dh_neg = get_dh(h_neg, h_q, d_sim_neg)

    # Backpropagation Through Time (BPTT)
    dWx_q, dWh_q, db_q = bptt(dh_q, hs_q, zs_q, xs_q, Wx, Wh)
    dWx_pos, dWh_pos, db_pos = bptt(dh_pos, hs_pos, zs_pos, xs_pos, Wx, Wh)
    dWx_neg, dWh_neg, db_neg = bptt(dh_neg, hs_neg, zs_neg, xs_neg, Wx, Wh)

    # Update Weights
    total_dWx = dWx_q + dWx_pos + dWx_neg
    total_dWh = dWh_q + dWh_pos + dWh_neg
    total_db  = db_q + db_pos + db_neg

    Wx -= LEARNING_RATE * (np.clip(total_dWx, -1.0, 1.0) + LAMBDA_REG * Wx)
    Wh -= LEARNING_RATE * (np.clip(total_dWh, -1.0, 1.0) + LAMBDA_REG * Wh)
    b  -= LEARNING_RATE * np.clip(total_db, -1.0, 1.0)

    return loss

# =============================================
#  PREDICT — single row, returns best label
# =============================================
def predict(row, vocab, idf, Wx, Wh, b):
    h_f    = rnn_encode(to_sequence(row["false_sent"], vocab, idf), Wx, Wh, b)
    scores = {
        l: cosine(h_f, rnn_encode(to_sequence(row[l], vocab, idf), Wx, Wh, b))
        for l in ["A", "B", "C"]
    }
    return max(scores, key=scores.get), scores

# =============================================
#  EVALUATE — accuracy on labelled split
# =============================================
def evaluate(data, vocab, idf, Wx, Wh, b):
    correct = sum(
        1 for row in data
        if predict(row, vocab, idf, Wx, Wh, b)[0] == row["answer"]
    )
    return correct / len(data) * 100

# =============================================
#  MAIN
# =============================================
if __name__ == "__main__":

    # ── load ─────────────────────────────────
    print("Loading data...")
    train_data = load_train(TRAIN_DATA_PATH, TRAIN_ANSWERS_PATH)
    test_data  = load_test(TEST_DATA_PATH)
    print(f"  Train: {len(train_data)} rows")
    print(f"  Test:  {len(test_data)} rows")

    # ── vocab + idf from ALL rows ─────────────
    print("\nBuilding vocab + IDF from all sentences...")
    vocab, idf = build_vocab_idf(train_data + test_data)
    print(f"  Vocab size: {len(vocab)}")

    # ── binary pairs from train only ─────────
    print("\nBuilding binary training pairs...")
    train = make_triplets(train_data, vocab, idf)
    # print(f"  {len(train)} pairs  "
    #       f"(pos={sum(1 for r in train if r['label']==1)}  "
    #       f"neg={sum(1 for r in train if r['label']==0)})")

    # ── initialise ────────────────────────────
    Wx, Wh, b = init_rnn(len(vocab))

    # ── training loop ─────────────────────────
    print(f"\n{'='*55}")
    print(f"  TRAINING  H={HIDDEN_DIM}  vocab={len(vocab)}")
    print(f"  lr={LEARNING_RATE}  L2={LAMBDA_REG}  epochs={EPOCHS}")
    print(f"{'='*55}")

    for epoch in range(EPOCHS):
        total_loss, active = 0.0, 0
        random.shuffle(train)

        for pair in train:
            # loss = train_step(pair["seq_s"], pair["seq_e"], pair["label"], Wx, Wh, b)
            loss = train_step_triplet(pair["q"], pair["pos"], pair["neg"], Wx, Wh, b)
            total_loss += loss
            if loss > 0:
                active += 1

        train_acc = evaluate(train_data, vocab, idf, Wx, Wh, b)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {total_loss/len(train):.4f} | "
              f"Active: {active}/{len(train)} | "
              f"Train acc: {train_acc:.2f}%")

    # ── predict on test set ───────────────────
    print(f"\nPredicting on {len(test_data)} test rows...", end=" ", flush=True)
    predictions = []
    for row in test_data:
        pred, _ = predict(row, vocab, idf, Wx, Wh, b)
        predictions.append({"id": row["id"], "answer": pred})
    print("done")

    # ── write predictions.csv ─────────────────
    with open(OUTPUT_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nPredictions written to: {OUTPUT_PATH}")
    print(f"  Rows: {len(predictions)}")
    print(f"\nSample:")
    for p in predictions[:5]:
        print(f"  {p['id']} → {p['answer']}")