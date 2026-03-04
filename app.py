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
 
    # สร้างตัวเลือกโดยหลีกเลี่ยงซ้ำกัน และไม่ให้ค่าซ้ำกับ correct เยอะ
    choices = set()
    choices.add(correct)
    while len(choices) < 4:
        delta = random.choice([1,2,3,4,5,6,7,8,9])
        sign = random.choice([1, -1])
        cand = correct + sign * delta
        if cand >= 0:
            choices.add(cand)
    choices = list(choices)
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
# แสดงข้อสอบ (ไม่มีตัวเลือกติ๊กไว้ตอนเริ่ม)
# =========================
scores = {"add":0, "sub":0, "mul":0, "div":0}
 
st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
st.caption("โปรดตอบทุกข้อ แล้วกดปุ่ม 'ส่งคำตอบ'")

# placeholder option (ใช้เป็นค่าเริ่มต้นที่ไม่ใช่คำตอบจริง)
PLACEHOLDER = "-- เลือกคำตอบ --"

# เก็บคำตอบของผู้ใช้ (ถ้ายังไม่มี ให้สร้างเป็น list ของ PLACEHOLDER)
if "user_answers" not in st.session_state:
    st.session_state.user_answers = [PLACEHOLDER] * len(st.session_state.questions)

# แสดงทีละข้อหรือทั้งหมด (คุณขอแสดงทั้งหมดก่อน — รักษาโครงเดิม)
for i, q in enumerate(st.session_state.questions):
    # สร้าง choices_with_placeholder แสดง placeholder เป็นตัวเลือกแรก
    choices_with_placeholder = [PLACEHOLDER] + [str(c) for c in q["choices"]]
    # เมื่อเลือกค่า ให้เก็บลง session_state.user_answers
    selected = st.radio(
        f"ข้อ {i+1}: {q['question']}",
        choices_with_placeholder,
        key=f"q{i}"
    )
    st.session_state.user_answers[i] = selected

# =========================
# ตรวจคำตอบ
# =========================
if st.button("ส่งคำตอบ"):
    # ตรวจว่าผู้ใช้ตอบครบทุกข้อ (ไม่มี placeholder)
    if any(ans == PLACEHOLDER for ans in st.session_state.user_answers):
        st.warning("กรุณาตอบให้ครบทุกข้อก่อนส่ง (ยังมีข้อที่ไม่ได้เลือกคำตอบ).")
    else:
        # นับคะแนน (q['answer'] เป็น int แต่ user_answers เก็บเป็น str -> แปลงเทียบ)
        for i, q in enumerate(st.session_state.questions):
            user_val = st.session_state.user_answers[i]
            try:
                # convert user selection to int (choices ถูกสร้างเป็นตัวเลข)
                user_val_num = int(user_val)
            except:
                user_val_num = None
            if user_val_num == q["answer"]:
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
    # เคลียร์ session keys ที่เราใช้
    keys_to_clear = ["questions", "user_answers"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # รีโหลดหน้า (ใช้ที่มีในเวอร์ชัน Streamlit ของคุณ)
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.script_request_rerun()
        except Exception:
            st.stop()
