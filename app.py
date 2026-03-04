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
        # สร้างตัวเลือกใกล้เคียง
        delta = random.choice([1,2,3,4,5,6,7,8,9])
        sign = random.choice([1, -1])
        cand = correct + sign * delta
        # ensure non-negative
        if cand >= 0:
            choices.add(cand)
    choices = list(choices)
    random.shuffle(choices)

    return question, correct, choices

# =========================
# เตรียมข้อสอบ 12 ข้อ (เก็บใน session_state)
# =========================
if "questions" not in st.session_state:
    st.session_state.questions = []
    # 3 ข้อต่อ operation ทั้ง 4 แบบ = 12 ข้อ
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
# สร้างตัวแปร session state สำหรับการแสดงทีละข้อ
# =========================
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

if "scores" not in st.session_state:
    st.session_state.scores = {"add": 0, "sub": 0, "mul": 0, "div": 0}

if "answered_flags" not in st.session_state:
    # เก็บว่าแต่ละข้อได้ถูกประเมินแล้วหรือยัง (ป้องกันเพิ่มคะแนนซ้ำ)
    st.session_state.answered_flags = [False] * len(st.session_state.questions)

if "selected_values" not in st.session_state:
    # เก็บคำตอบที่เลือกไว้ต่อข้อ (เพื่อแสดงค่าที่เลือกเมื่อ rerun)
    st.session_state.selected_values = [None] * len(st.session_state.questions)

if "finished" not in st.session_state:
    st.session_state.finished = False

# หัวข้อ / สถานะ
st.subheader("📘 ทำแบบทดสอบ 12 ข้อ — ทีละข้อ")
st.caption("ตอบคำถามแล้วกด 'ยืนยัน' เพื่อเช็คคำตอบ แล้วกด 'ถัดไป' เพื่อไปข้อถัดไป")

# =========================
# แสดงทีละข้อ (หลัก)
# =========================
if not st.session_state.finished and st.session_state.current_idx < len(st.session_state.questions):
    idx = st.session_state.current_idx
    qdata = st.session_state.questions[idx]

    st.write(f"### ข้อที่ {idx+1} / {len(st.session_state.questions)}")
    st.write(qdata["question"])

    # แสดง radio โดยใช้ key เฉพาะของข้อ
    radio_key = f"q_radio_{idx}"
    # ถ้ามีค่าเก่าบันทึกไว้ ให้กำหนด value อัตโนมัติ (st.session_state.selected_values ถูกใช้)
    if st.session_state.selected_values[idx] is None:
        # no default selection
        selected = st.radio("", qdata["choices"], key=radio_key)
        st.session_state.selected_values[idx] = selected
    else:
        # ตั้ง default ให้กับ radio โดยอ่านจาก session_state
        try:
            selected = st.radio("", qdata["choices"], index=qdata["choices"].index(st.session_state.selected_values[idx]), key=radio_key)
        except Exception:
            # หากค่าเดิมไม่ตรงตัวเลือกใด (edge case) ให้เลือกตัวแรก
            selected = st.radio("", qdata["choices"], key=radio_key)
            st.session_state.selected_values[idx] = selected

    # ปุ่มยืนยันคำตอบ (ประเมินคำตอบครั้งเดียว)
    if not st.session_state.answered_flags[idx]:
        if st.button("ยืนยันคำตอบ", key=f"confirm_{idx}"):
            # ตรวจคำตอบ — เพิ่มคะแนนเฉพาะครั้งแรก
            if selected == qdata["answer"]:
                st.success("✅ ถูกต้อง")
                # เพิ่มคะแนนใน scores ครั้งเดียว
                if qdata["operation"] == "add":
                    st.session_state.scores["add"] += 1
                elif qdata["operation"] == "sub":
                    st.session_state.scores["sub"] += 1
                elif qdata["operation"] == "mul":
                    st.session_state.scores["mul"] += 1
                elif qdata["operation"] == "div":
                    st.session_state.scores["div"] += 1
            else:
                st.error(f"❌ ผิด — เฉลย: {qdata['answer']}")

            st.session_state.answered_flags[idx] = True
            # ให้ผู้ใช้เห็นผลก่อนจะกดถัดไป — ไม่ rerun อัตโนมัติ เพื่อให้ข้อความแสดง
    else:
        # ถ้าประเมินแล้ว ให้แสดงปุ่มถัดไป (หรือจบหากเป็นข้อสุดท้าย)
        if idx < len(st.session_state.questions) - 1:
            if st.button("ถัดไป", key=f"next_{idx}"):
                st.session_state.current_idx += 1
                st.experimental_rerun()
        else:
            # ข้อสุดท้าย — ให้ปุ่มจบ/สรุป
            if st.button("จบและดูผล"):
                st.session_state.finished = True
                st.experimental_rerun()

# =========================
# แสดงผลเมื่อทำครบ (สรุป และส่งเข้า ML)
# =========================
if st.session_state.finished or st.session_state.current_idx >= len(st.session_state.questions):
    # คำนวณคะแนนแต่ละหมวด (จากจำนวนที่ถูก / 3 ข้อต่อหมวด)
    add_score = round((st.session_state.scores["add"] / 3) * 100, 2)
    sub_score = round((st.session_state.scores["sub"] / 3) * 100, 2)
    mul_score = round((st.session_state.scores["mul"] / 3) * 100, 2)
    div_score = round((st.session_state.scores["div"] / 3) * 100, 2)

    st.subheader("📊 ผลคะแนน (ต่อหมวด)")
    st.write(f"➕ การบวก: {add_score} / 100")
    st.write(f"➖ การลบ: {sub_score} / 100")
    st.write(f"✖ การคูณ: {mul_score} / 100")
    st.write(f"➗ การหาร: {div_score} / 100")

    # ถ้าได้เต็มทุกหมวด
    if add_score == 100 and sub_score == 100 and mul_score == 100 and div_score == 100:
        st.success("🎉 คุณพร้อมเรียนบทต่อไปแล้ว!")
    else:
        # ส่งเข้า ML เพื่อตรวจจุดอ่อน
        prediction = model.predict([[add_score, sub_score, mul_score, div_score]])
        result = prediction[0]

        st.subheader("🤖 ผลการวิเคราะห์จาก AI")
        st.info(f"จุดที่ควรพัฒนา: {result}")

        # แนะนำคลิป (ตัวอย่าง)
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
# ปุ่มเริ่มใหม่ (คงโครงเดิม)
# =========================
st.markdown("---")
if st.button("🔄 เริ่มใหม่ทั้งหมด"):
    # เคลียร์ทุก session state ที่เราใช้ (แต่ไม่ลบโค้ดต้นทาง)
    keys_to_keep = []  # ถ้าต้องการเก็บอะไรไว้ให้ใส่ชื่อคีย์นี้
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep:
            del st.session_state[k]
    st.experimental_rerun()
