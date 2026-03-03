import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Learning Competition", layout="centered")

st.title("🏆 AI ระบบแนะนำการเรียนแบบแข่งขัน")

# =========================
# โหลดและเทรน ML (Cache ครั้งเดียว)
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    df["average"] = (df["math score"] +
                     df["reading score"] +
                     df["writing score"]) / 3

    def label_level(avg):
        if avg < 50:
            return 0
        elif avg < 75:
            return 1
        else:
            return 2

    df["level"] = df["average"].apply(label_level)

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
# คลังข้อสอบ 3 วิชา
# =========================
question_bank = {
    "Math": {
        1: [
            {"q": "2 + 3 =", "c": ["4", "5", "6"], "a": "5"},
            {"q": "5 × 2 =", "c": ["10", "12", "8"], "a": "10"},
            {"q": "10 ÷ 2 =", "c": ["3", "5", "8"], "a": "5"},
            {"q": "3 + 7 =", "c": ["9", "10", "8"], "a": "10"},
            {"q": "6 - 1 =", "c": ["4", "5", "6"], "a": "5"},
        ],
        2: [
            {"q": "x + 5 = 9, x =", "c": ["3", "4", "5"], "a": "4"},
            {"q": "12 ÷ 3 =", "c": ["4", "3", "6"], "a": "4"},
            {"q": "7 × 3 =", "c": ["21", "24", "18"], "a": "21"},
            {"q": "15 - 6 =", "c": ["9", "8", "10"], "a": "9"},
            {"q": "9 + 8 =", "c": ["16", "17", "18"], "a": "17"},
        ],
        3: [
            {"q": "อนุพันธ์ของ x² =", "c": ["2x", "x", "x²"], "a": "2x"},
            {"q": "∫ 2x dx =", "c": ["x² + C", "2x + C", "x + C"], "a": "x² + C"},
            {"q": "lim x→0 (x²/x) =", "c": ["0", "1", "∞"], "a": "0"},
            {"q": "2² + 3² =", "c": ["12", "13", "14"], "a": "13"},
            {"q": "√49 =", "c": ["6", "7", "8"], "a": "7"},
        ]
    },
    "Thai": {
        1: [
            {"q": "คำราชาศัพท์ของ กิน คือ", "c": ["เสวย", "รับประทาน", "กิน"], "a": "เสวย"},
            {"q": "คำพ้องรูปคือ", "c": ["คำเหมือน", "เขียนเหมือน", "เสียงเหมือน"], "a": "เขียนเหมือน"},
            {"q": "สระในคำว่า บ้าน คือ", "c": ["า", "บ", "น"], "a": "า"},
            {"q": "คำว่า แมว เป็น", "c": ["คำนาม", "คำกริยา", "คำวิเศษณ์"], "a": "คำนาม"},
            {"q": "คำตรงข้ามของ ดี", "c": ["เก่ง", "เลว", "ขาว"], "a": "เลว"},
        ],
        2: [
            {"q": "ประโยคความเดียวคือ", "c": ["มีประธานเดียว", "มี 2 ประโยค", "มี 3 กริยา"], "a": "มีประธานเดียว"},
            {"q": "คำบุพบทคือ", "c": ["ใน", "กิน", "ดี"], "a": "ใน"},
            {"q": "คำวิเศษณ์คือ", "c": ["เร็ว", "แมว", "กิน"], "a": "เร็ว"},
            {"q": "อักษรควบแท้คือ", "c": ["กร", "ทร", "สร"], "a": "กร"},
            {"q": "คำซ้อนคือ", "c": ["สวยงาม", "แดง", "กิน"], "a": "สวยงาม"},
        ],
        3: [
            {"q": "กลอนสุภาพมีกี่วรรค", "c": ["2", "3", "4"], "a": "4"},
            {"q": "โวหารเปรียบเทียบคือ", "c": ["อุปมา", "บรรยาย", "พรรณนา"], "a": "อุปมา"},
            {"q": "คำสมาสคือ", "c": ["ราชการ", "กินข้าว", "เดินเล่น"], "a": "ราชการ"},
            {"q": "ฉันทลักษณ์หมายถึง", "c": ["กฎแต่งคำประพันธ์", "คำสวย", "เสียงเพราะ"], "a": "กฎแต่งคำประพันธ์"},
            {"q": "คำอุปไมยคือ", "c": ["เปรียบโดยไม่มีคำว่าเหมือน", "เหมือน", "คล้าย"], "a": "เปรียบโดยไม่มีคำว่าเหมือน"},
        ]
    },
    "English": {
        1: [
            {"q": "He ___ a student.", "c": ["is", "are", "am"], "a": "is"},
            {"q": "I ___ to school.", "c": ["go", "goes", "went"], "a": "go"},
            {"q": "Cat is a", "c": ["animal", "fruit", "car"], "a": "animal"},
            {"q": "They ___ happy.", "c": ["is", "are", "am"], "a": "are"},
            {"q": "She ___ coffee.", "c": ["like", "likes", "liked"], "a": "likes"},
        ],
        2: [
            {"q": "Past of go", "c": ["gone", "went", "goed"], "a": "went"},
            {"q": "Synonym of big", "c": ["large", "small", "short"], "a": "large"},
            {"q": "Plural of child", "c": ["childs", "children", "childes"], "a": "children"},
            {"q": "He has ___ book.", "c": ["a", "an", "the"], "a": "a"},
            {"q": "Opposite of hot", "c": ["cold", "warm", "heat"], "a": "cold"},
        ],
        3: [
            {"q": "If I ___ rich, I would travel.", "c": ["am", "were", "was"], "a": "were"},
            {"q": "Passive of 'They build a house'", "c": ["A house is built", "A house built", "Built house"], "a": "A house is built"},
            {"q": "Reported speech of 'I am tired'", "c": ["He said he was tired", "He said I am tired", "He tired"], "a": "He said he was tired"},
            {"q": "Gerund is", "c": ["Verb+ing as noun", "Past verb", "Adjective"], "a": "Verb+ing as noun"},
            {"q": "Present perfect form", "c": ["have + V3", "had + V2", "is + V1"], "a": "have + V3"},
        ]
    }
}

# =========================
# Session State
# =========================
if "started" not in st.session_state:
    st.session_state.started = False

if "level" not in st.session_state:
    st.session_state.level = 1

if "score" not in st.session_state:
    st.session_state.score = 0

if "streak" not in st.session_state:
    st.session_state.streak = 0

if "count" not in st.session_state:
    st.session_state.count = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = None

# =========================
# เลือกวิชา
# =========================
subject = st.selectbox("เลือกวิชา", ["Math", "Thai", "English"])

# =========================
# เริ่มเกม
# =========================
if st.button("เริ่มแข่งขันใหม่"):
    st.session_state.started = True
    st.session_state.level = 1
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.count = 0
    st.session_state.current_question = None

# =========================
# ระบบทำข้อสอบ
# =========================
if st.session_state.started and st.session_state.count < 5:

    if st.session_state.current_question is None:
        st.session_state.current_question = random.choice(
            question_bank[subject][st.session_state.level]
        )

    q = st.session_state.current_question

    st.subheader(f"Level {st.session_state.level}")
    st.write(q["q"])

    choice = st.radio("เลือกคำตอบ:", q["c"], key="choice")

    if st.button("ส่งคำตอบ"):
        if choice == q["a"]:
            st.success("ถูกต้อง ✅")
            st.session_state.score += 1
            st.session_state.streak += 1
        else:
            st.error("ผิด ❌")
            st.session_state.streak -= 1

        # Adaptive
        if st.session_state.streak == 2:
            st.session_state.level = min(3, st.session_state.level + 1)
            st.session_state.streak = 0

        if st.session_state.streak == -2:
            st.session_state.level = max(1, st.session_state.level - 1)
            st.session_state.streak = 0

        st.session_state.count += 1
        st.session_state.current_question = None

# =========================
# จบรอบ → ML ทำนาย
# =========================
if st.session_state.count == 5:
    st.success(f"คะแนนรวม: {st.session_state.score}/5")

    score_100 = (st.session_state.score / 5) * 100

    if subject == "Math":
        input_data = [[score_100, 60, 60]]
    elif subject == "Thai":
        input_data = [[60, score_100, 60]]
    else:
        input_data = [[60, 60, score_100]]

    pred = model.predict(input_data)[0]

    level_map = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    predicted_level = level_map[pred]

    st.subheader(f"🎯 ระดับที่ AI ทำนาย: {predicted_level}")

    if predicted_level == "Beginner":
        st.write("แนะนำทบทวนพื้นฐานเพิ่มเติม")
    elif predicted_level == "Intermediate":
        st.write("ควรฝึกโจทย์ประยุกต์เพิ่ม")
    else:
        st.write("พร้อมสำหรับโจทย์ขั้นสูง")

    if st.button("เริ่มใหม่อีกครั้ง"):
        st.session_state.started = False
