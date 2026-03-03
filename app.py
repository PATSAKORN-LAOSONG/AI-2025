import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Learning Competition", layout="centered")

st.title("🏆 AI ระบบแนะนำการเรียนแบบแข่งขัน")

# =========================
# 1️⃣ Train ML (Cache ครั้งเดียว)
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
# 2️⃣ คลังข้อสอบ
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
# 3️⃣ Session State
# =========================
defaults = {
    "started": False,
    "level": 1,
    "score": 0,
    "streak": 0,
    "count": 0,
    "current_question": None,
    "question_id": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================
# 4️⃣ เลือกวิชา
# =========================
subject = st.selectbox("เลือกวิชา", ["Math"])

# =========================
# 5️⃣ ปุ่มเริ่มใหม่
# =========================
if st.button("เริ่มแข่งขันใหม่"):
    for key in defaults:
        st.session_state[key] = defaults[key]

    st.session_state.started = True
    st.session_state.question_id = random.randint(1, 100000)
    st.rerun()

# =========================
# 6️⃣ ระบบทำข้อสอบ
# =========================
if st.session_state.started and st.session_state.count < 5:

    if st.session_state.current_question is None:
        st.session_state.current_question = random.choice(
            question_bank[subject][st.session_state.level]
        )
        st.session_state.question_id = random.randint(1, 100000)

    q = st.session_state.current_question

    st.write(f"ข้อที่ {st.session_state.count + 1} / 5")
    st.write(q["q"])

    choice = st.radio(
        "เลือกคำตอบ:",
        q["c"],
        key=f"choice_{st.session_state.question_id}"
    )

    if st.button("ส่งคำตอบ"):

        if choice == q["a"]:
            st.success("ถูกต้อง ✅")
            st.session_state.score += 1
            st.session_state.streak += 1
        else:
            st.error("ผิด ❌")
            st.session_state.streak -= 1

        # Adaptive logic
        if st.session_state.streak == 2:
            st.session_state.level = min(3, st.session_state.level + 1)
            st.session_state.streak = 0

        if st.session_state.streak == -2:
            st.session_state.level = max(1, st.session_state.level - 1)
            st.session_state.streak = 0

        st.session_state.count += 1
        st.session_state.current_question = None

        st.rerun()

# =========================
# 7️⃣ จบรอบ → ทำนาย ML
# =========================
if st.session_state.count == 5:

    st.success(f"คะแนนรวม: {st.session_state.score}/5")

    score_100 = (st.session_state.score / 5) * 100

    input_data = [[score_100, 60, 60]]

    pred = model.predict(input_data)[0]

    level_map = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    predicted_level = level_map[pred]

    st.subheader(f"🎯 AI ประเมินระดับคุณ: {predicted_level}")

    if predicted_level == "Beginner":
        st.write("ควรทบทวนพื้นฐานเพิ่มเติม")
    elif predicted_level == "Intermediate":
        st.write("ควรฝึกโจทย์ประยุกต์เพิ่ม")
    else:
        st.write("พร้อมทำโจทย์ขั้นสูงได้")

    if st.button("เริ่มใหม่อีกครั้ง"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()
