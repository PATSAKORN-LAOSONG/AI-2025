import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 
st.set_page_config(page_title="Math Skill AI", layout="centered")
 
st.title("🧠 ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์ด้วย AI")
 
# =========================
# โหลด Dataset และ Train ML
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("math_skill_dataset_200.csv")
    X = df[["addition", "subtraction", "multiplication", "division"]]
    y = df["label"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
 
model = train_model()
 
# =========================
# ฟังก์ชันสร้างโจทย์
# =========================
def generate_question(operation):
    if operation == "add":
        a, b = random.randint(1, 50), random.randint(1, 50)
        correct = a + b
        question = f"{a} + {b} = ?"
 
    elif operation == "sub":
        a, b = random.randint(1, 50), random.randint(1, 50)
        if a < b:
            a, b = b, a
        correct = a - b
        question = f"{a} - {b} = ?"
 
    elif operation == "mul":
        a, b = random.randint(1, 12), random.randint(1, 12)
        correct = a * b
        question = f"{a} × {b} = ?"
 
    elif operation == "div":
        b = random.randint(1, 12)
        correct = random.randint(1, 12)
        a = b * correct
        question = f"{a} ÷ {b} = ?"
 
    choices = [correct,
               correct + random.randint(1, 5),
               correct - random.randint(1, 5),
               correct + random.randint(6, 10)]
 
    random.shuffle(choices)
 
    return question, correct, choices
 
# =========================
# เตรียมข้อสอบ 12 ข้อ
# =========================
if "questions" not in st.session_state:
    st.session_state.questions = []
    operations = ["add"]*3 + ["sub"]*3 + ["mul"]*3 + ["div"]*3
    random.shuffle(operations)
 
    for op in operations:
        q, ans, choices = generate_question(op)
        st.session_state.questions.append({
            "operation": op,
            "question": q,
            "answer": ans,
            "choices": choices
        })
 
# =========================
# แสดงข้อสอบ
# =========================
scores = {"add":0, "sub":0, "mul":0, "div":0}
 
st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
 
user_answers = []
 
for i, q in enumerate(st.session_state.questions):
    user_choice = st.radio(
        f"ข้อ {i+1}: {q['question']}",
        q["choices"],
        key=f"q{i}"
    )
    user_answers.append(user_choice)
 
# =========================
# ตรวจคำตอบ
# =========================
if st.button("ส่งคำตอบ"):
    for i, q in enumerate(st.session_state.questions):
        if user_answers[i] == q["answer"]:
            scores[q["operation"]] += 1
 
    # Normalize 0–100
    add_score = round((scores["add"]/3)*100,2)
    sub_score = round((scores["sub"]/3)*100,2)
    mul_score = round((scores["mul"]/3)*100,2)
    div_score = round((scores["div"]/3)*100,2)
 
    st.subheader("📊 ผลคะแนน")
    st.write(f"➕ การบวก: {add_score}")
    st.write(f"➖ การลบ: {sub_score}")
    st.write(f"✖ การคูณ: {mul_score}")
    st.write(f"➗ การหาร: {div_score}")
 
    # ถ้าได้เต็มทุกหมวด
    if add_score == 100 and sub_score == 100 and mul_score == 100 and div_score == 100:
        st.success("🎉 คุณพร้อมเรียนบทต่อไปแล้ว!")
    else:
        # ส่งเข้า ML
        prediction = model.predict([[add_score, sub_score, mul_score, div_score]])
        result = prediction[0]
 
        st.subheader("🤖 ผลการวิเคราะห์จาก AI")
        st.info(f"จุดที่ควรพัฒนา: {result}")
 
        # แนะนำคลิป
        if result == "weak_add":
            st.write("แนะนำฝึกการบวกเพิ่มเติม:")
            st.video("https://www.youtube.com/watch?v=c5eS7nRsE_Q")
 
        elif result == "weak_sub":
            st.write("แนะนำฝึกการลบเพิ่มเติม:")
            st.video("https://www.youtube.com/watch?v=vT_VBLlvdn8")
 
        elif result == "weak_mul":
            st.write("แนะนำฝึกการคูณเพิ่มเติม:")
            st.video("https://www.youtube.com/watch?v=73obrcsERe8")
 
        elif result == "weak_div":
            st.write("แนะนำฝึกการหารเพิ่มเติม:")
            st.video("https://www.youtube.com/watch?v=9D1JW8rYqeA")
 
        elif result == "strong_all":
            st.success("คุณมีพื้นฐานแข็งแรงทุกด้าน 👍")
 
# =========================
# ปุ่มเริ่มใหม่
# =========================
if st.button("🔄 เริ่มใหม่"):
    st.session_state.clear()
    st.rerun()
