import json
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Pembacaan dataset jsonl
dataset_file = "indonesian-dev.jsonl"  

if not os.path.exists(dataset_file):
    print(f"⚠️ File {dataset_file} tidak ditemukan!")
    exit()

questions = []
answers = []

#Membaca dataset jsonl
with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        question = data.get("question_text", "").strip()

        # Ambil jawaban dari minimal_answer
        answer_data = data.get("annotations", [{}])[0]
        answer = answer_data.get("minimal_answer", {}).get("text", "").strip()

        # Jika tidak ada jawaban, coba ambil dari passage_answer_candidates
        if not answer and "passage_answer_candidates" in data:
            candidates = data["passage_answer_candidates"]
            if candidates:
                start = candidates[0].get("plaintext_start_byte", 0)
                end = candidates[0].get("plaintext_end_byte", len(data["document_plaintext"]))
                answer = data["document_plaintext"][start:end].strip()

        if not answer:
            answer = "(Jawaban tidak tersedia)"

        questions.append(question)
        answers.append(answer)

# Mengecek jawaban 
valid_answers = [ans for ans in answers if ans != "(Jawaban tidak tersedia)"]

if len(valid_answers) == 0:
    print("❌ Semua jawaban dalam dataset kosong! Pastikan dataset memiliki jawaban yang valid.")
    exit()

print(f"✅ Dataset dimuat! Total pertanyaan: {len(questions)}")
print(f"📊 Total jawaban valid: {len(valid_answers)} dari {len(answers)}")

# 2. NLP: Konversi teks ke angka dengan TF-IDF 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)  # Konversi pertanyaan ke vektor
y = np.arange(len(questions))  # Label berdasarkan indeks jawaban

# 3. melatih Model SVM 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear")
model.fit(X_train, y_train)
print("✅ Model SVM dilatih!")

# 4. Fungsi Evaluasi Jawaban User 
def evaluate_answer(user_question, user_answer):
    user_vector = vectorizer.transform([user_question])
    pred = model.predict(user_vector)

    # Ambil jawaban referensi dengan aman
    if 0 <= pred[0] < len(answers):  
        correct_answer = answers[pred[0]].strip()
    else:
        correct_answer = "(Jawaban tidak ditemukan)"

    # Jika jawaban referensi tidak tersedia
    if not correct_answer or correct_answer == "(Jawaban tidak tersedia)":
        return 0, "Jawaban referensi tidak tersedia dalam dataset.", correct_answer

    # Pastikan jawaban pengguna tidak kosong
    if not user_answer.strip():
        return 0, "Jawaban tidak boleh kosong!", correct_answer

    # Hitung kemiripan dengan jawaban referensi
    user_answer_vector = vectorizer.transform([user_answer])
    correct_answer_vector = vectorizer.transform([correct_answer])
    similarity = cosine_similarity(user_answer_vector, correct_answer_vector)[0][0]

    score = int(similarity * 100)

    # Update feedback berdasarkan skor
    if score >= 90:
        feedback = "Jawaban sangat baik!"
    elif score >= 70:
        feedback = "Jawaban cukup baik, coba lebih detail."
    elif score >= 50:
        feedback = "Jawaban masih perlu dikembangkan."
    elif score >= 30:
        feedback = "Jawaban kurang relevan, coba lebih sesuai."
    else:
        feedback = "Jawaban tidak sesuai."

    return score, feedback, correct_answer



# 5. Web Interface dengan Flask 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_question = request.form.get("question", "").strip()
        user_answer = request.form.get("answer", "").strip()
        
        if not user_question or not user_answer:
            return render_template("index.html", question=user_question, answer=user_answer,
                                   score=None, feedback="Harap isi pertanyaan dan jawaban!", correct_answer=None)
        
        score, feedback, correct_answer = evaluate_answer(user_question, user_answer)

        return render_template("index.html", question=user_question, answer=user_answer,
                               score=score, feedback=feedback, correct_answer=correct_answer)

    return render_template("index.html", question="", answer="", score=None, feedback=None, correct_answer=None)

if __name__ == "__main__":
    app.run(debug=True)
