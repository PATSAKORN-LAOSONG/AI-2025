import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Learning Recommendation System", layout="centered")
st.title("🤖 AI ระบบแนะนำการเรียน (Adaptive + ML)")

# ==========================================
# 1️⃣ TRAIN MODEL (CACHE ครั้งเดียว)
# ==========================================
@st.cache_resource
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    df["average"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    ) / 3

    def create_level(avg):
        if avg < 50:
            return "Beginner"
        elif avg <= 75:
            return "Intermediate"
        else:
            return "Advanced"

    df["level"] = df["average"].apply(create_level)

    X = df[["math score", "reading score", "writing score"]]
    y = df["level"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

model = train_model()

# ==========================================
# 2️⃣ SESSION STATE
# ==========================================
defaults = {
    "level": "medium",
    "score": 0,
    "question_count": 0,
    "correct_streak": 0,
    "wrong_streak": 0,
    "finished": False,
    "subject_scores": {
        "math": 0,
        "reading": 0,
        "writing": 0
    }
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

levels = ["easy", "medium", "hard"]

# ==========================================
# 3️⃣ QUESTION BANK (5 ข้อต่อระดับ)
# ==========================================
question_bank = {
    "math": {
        "easy": [
            {"q":"2+3=?","choices":["4","5","6","7"],"answer":"5"},
            {"q":"4+4=?","choices":["6","7","8","9"],"answer":"8"},
            {"q":"6-1=?","choices":["4","5","6","7"],"answer":"5"},
            {"q":"3x2=?","choices":["5","6","7","8"],"answer":"6"},
            {"q":"10÷2=?","choices":["3","4","5","6"],"answer":"5"},
        ],
        "medium": [
            {"q":"5^2=?","choices":["10","20","25","15"],"answer":"25"},
            {"q":"√16=?","choices":["2","3","4","5"],"answer":"4"},
            {"q":"12/3=?","choices":["2","3","4","5"],"answer":"4"},
            {"q":"7+8=?","choices":["14","15","16","17"],"answer":"15"},
            {"q":"9-3=?","choices":["5","6","7","8"],"answer":"6"},
        ],
        "hard": [
            {"q":"อนุพันธ์ x^2=?","choices":["x","2x","x^2","2"],"answer":"2x"},
            {"q":"อินทิกรัล x^2=?","choices":["x^3/3","x^2","2x","3x"],"answer":"x^3/3"},
            {"q":"sin(90)=?","choices":["0","1","-1","0.5"],"answer":"1"},
            {"q":"log10(100)=?","choices":["1","2","10","100"],"answer":"2"},
            {"q":"3^3=?","choices":["6","9","27","81"],"answer":"27"},
        ]
    }
}

subject = "math"

# ==========================================
# 4️⃣ QUIZ SYSTEM
# ==========================================
if not st.session_state.finished:

    q = random.choice(question_bank[subject][st.session_state.level])

    st.write(f"ข้อที่ {st.session_state.question_count+1} (ระดับ {st.session_state.level})")
    st.write(q["q"])

    ans = st.radio("เลือกคำตอบ", q["choices"])

    if st.button("ส่งคำตอบ"):

        if ans == q["answer"]:
            st.session_state.score += 1
            st.session_state.correct_streak += 1
            st.session_state.wrong_streak = 0
        else:
            st.session_state.wrong_streak += 1
            st.session_state.correct_streak = 0

        if st.session_state.correct_streak == 2:
            if st.session_state.level != "hard":
                st.session_state.level = levels[levels.index(st.session_state.level)+1]
            st.session_state.correct_streak = 0

        if st.session_state.wrong_streak == 2:
            if st.session_state.level != "easy":
                st.session_state.level = levels[levels.index(st.session_state.level)-1]
            st.session_state.wrong_streak = 0

        st.session_state.question_count += 1

        if st.session_state.question_count >= 5:
            st.session_state.finished = True

        st.rerun()

# ==========================================
# 5️⃣ RESULT + ML PREDICTION
# ==========================================
if st.session_state.finished:

    st.success(f"คะแนนรวม: {st.session_state.score}/5")

    scaled_score = (st.session_state.score / 5) * 100

    prediction = model.predict([[scaled_score, scaled_score, scaled_score]])[0]

    st.info(f"🤖 AI ประเมินระดับคุณเป็น: {prediction}")

    if prediction == "Beginner":
        st.write("📘 แนะนำทบทวนพื้นฐานเพิ่มเติม")
    elif prediction == "Intermediate":
        st.write("📗 คุณอยู่ระดับกลาง ฝึกต่อเนื่อง")
    else:
        st.write("📕 คุณอยู่ระดับสูง สามารถทำโจทย์ขั้นสูงได้")

    if st.button("เริ่มใหม่"):
        for k in defaults.keys():
            st.session_state[k] = defaults[k]
        st.rerun()
