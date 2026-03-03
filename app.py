import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Learning Competition", layout="centered")
st.title("🏆 AI ระบบแนะนำการเรียนแบบแข่งขัน")

# =========================
# 1️⃣ Train ML (Cache)
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    df["average"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    ) / 3

    def label(avg):
        if avg < 50:
            return 0
        elif avg < 75:
            return 1
        else:
            return 2

    df["level"] = df["average"].apply(label)

    X = df[["math score", "reading score", "writing score"]]
    y = df["level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model

model = train_model()

# =========================
# 2️⃣ Question Bank
# =========================
question_bank = {
    "Math": {
        1: [
            {"q": "2 + 3 =", "c": ["4", "5", "6"], "a": "5"},
            {"q": "3 + 7 =", "c": ["9", "10", "8"], "a": "10"},
            {"q": "6 - 1 =", "c": ["4", "5", "6"], "a": "5"},
        ],
        2: [
            {"q": "7 × 3 =", "c": ["21", "24", "18"], "a": "21"},
            {"q": "15 - 6 =", "c": ["9", "8", "10"], "a": "9"},
            {"q": "12 ÷ 3 =", "c": ["4", "3", "6"], "a": "4"},
        ],
        3: [
            {"q": "อนุพันธ์ของ x² =", "c": ["2x", "x", "x²"], "a": "2x"},
            {"q": "√49 =", "c": ["6", "7", "8"], "a": "7"},
            {"q": "2² + 3² =", "c": ["12", "13", "14"], "a": "13"},
        ]
    }
}

# =========================
# 3️⃣ Session Defaults
# =========================
defaults = {
    "started": False,
    "level": 1,
    "score": 0,
    "streak": 0,
    "count": 0,
    "questions_pool": [],
    "current_question": None,
    "question_id": 0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

subject = st.selectbox("เลือกวิชา", ["Math"])

# =========================
# 4️⃣ Start Button
# =========================
if st.button("เริ่มแข่งขันใหม่"):
    for k in defaults:
        st.session_state[k] = defaults[k]

    st.session_state.started = True

    st.session_state.questions_pool = random.sample(
        question_bank[subject][1],
        len(question_bank[subject][1])
    )

    st.session_state.current_question = None
    st.session_state.question_id = random.randint(1, 100000)

    st.rerun()

# =========================
# 5️⃣ Quiz System (FIXED)
# =========================
if st.session_state.started and st.session_state.count < 5:

    # 🔥 สุ่มเฉพาะตอนยังไม่มีคำถามปัจจุบัน
    if st.session_state.current_question is None:

        if len(st.session_state.questions_pool) == 0:
            st.session_state.questions_pool = random.sample(
                question_bank[subject][st.session_state.level],
                len(question_bank[subject][st.session_state.level])
            )

        st.session_state.current_question = st.session_state.questions_pool[0]

    q = st.session_state.current_question

    st.write(f"ข้อที่ {st.session_state.count + 1} / 5")
    st.write(q["q"])

    choice = st.radio(
        "เลือกคำตอบ:",
        q["c"],
        key=f"choice_{st.session_state.question_id}"
    )

    if st.button("ส่งคำตอบ"):

        # ตรวจคำตอบ
        if choice == q["a"]:
            st.success("ถูกต้อง ✅")
            st.session_state.score += 1
            st.session_state.streak += 1
        else:
            st.error("ผิด ❌")
            st.session_state.streak -= 1

        # ลบข้อที่ทำแล้ว (pop ตอนนี้เท่านั้น)
        st.session_state.questions_pool.pop(0)

        # Adaptive
        if st.session_state.streak == 2:
            st.session_state.level = min(3, st.session_state.level + 1)
            st.session_state.streak = 0
            st.session_state.questions_pool = []

        if st.session_state.streak == -2:
            st.session_state.level = max(1, st.session_state.level - 1)
            st.session_state.streak = 0
            st.session_state.questions_pool = []

        st.session_state.count += 1
        st.session_state.current_question = None
        st.session_state.question_id = random.randint(1, 100000)

        st.rerun()

# =========================
# 6️⃣ ML Prediction
# =========================
if st.session_state.count == 5:

    st.success(f"คะแนนรวม: {st.session_state.score}/5")

    score_100 = (st.session_state.score / 5) * 100
    input_data = [[score_100, 60, 60]]

    pred = model.predict(input_data)[0]
    level_map = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}

    st.subheader(f"🎯 AI ประเมินระดับคุณ: {level_map[pred]}")

    if st.button("เริ่มใหม่อีกครั้ง"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()
