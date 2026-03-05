# ไลบรารีสำหรับสร้างเว็บแอปจาก Python
import streamlit as st
# ไลบรารีสำหรับใช้สำหรับสุ่มค่า
import random
# ไลบรารีสำหรับจัดการ ข้อมูลตาราง CSV
import pandas as pd
# ไลบรารีโมเดล Random Forest
from sklearn.ensemble import RandomForestClassifier
# ไลบรารีใช้สำหรับแบ่งข้อมูลไว้ Train กับ Test
from sklearn.model_selection import train_test_split
# เป็นการตั้งค่าหน้าเว็ป(ชื่อบนบราวเชอร์,ทำให้เนื้อหาในเว็ปอยู่ตรงกลางจอ)
st.set_page_config(page_title="ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์", layout="centered")
# สร้างหัวข้อใหญ่หน้าเว็ป
st.title("🧠 ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์ด้วย AI")

# =========================
# โหลด CSV และ Train ML
# =========================
# ใช้เพื่อแคชข้อมูลหรือเก็บผลลัพธ์ไว้ชั่วคราว
@st.cache_resource
# การสร้างฟังก์ชัน
def train_model():
# โหลดไฟล์ เข้าตัวแปร df
    df = pd.read_csv("math_skill_dataset_200.csv")
    # แยก Feature หรือข้อมูลที่ใช้ทำนาย มีทักษะ บวก ลบ คูณ หาร
    X = df[["addition", "subtraction", "multiplication", "division"]]
    # แยก Label สิ่งที่จะให้ Ai ทำนาย จะมี (stong_all,weak_add,weak_sub,weak_mul,weak_div)
    y = df["label"]
    # แบ่งข้อมูล Train / Test = 80% / 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # สร้างโมเดล Random Forest ส่วน random_state=42 คือการกำหนด seed เพื่อให้การแบ่งข้อมูลเหมือนเดิมทุกครั้ง
    model = RandomForestClassifier(random_state=42)
    # ฝึกโมเดลจากข้อมูล Train
    model.fit(X_train, y_train)
    # ส่งค่า model กับ dataset
    return model, df
    # เรียกค่า model กับ df มาใส่ใน train_model()
model, df = train_model()

# =========================
# การสร้างโจทย์
# =========================
# สร้างฟังก์ชั่นชื่อ generate_question โดยรับค่า operation ที่บอกว่าเป็นโจทย์ของทักษะอะไร
def generate_question(operation):
## โจทย์บวก
    if operation == "add":
        # มีการสุ่มเลข 1-50 ในตัวแปร a และ b
        a, b = random.randint(1, 50), random.randint(1, 50)
        # คำนวนคำตอบที่ถูก
        correct = a + b
        # สรา้งข้อความโจทย์
        question = f"{a} + {b} = ?"
## โจทย์ลบ
    elif operation == "sub":
        a, b = random.randint(1, 50), random.randint(1, 50)
        # ทำการสลับค่าถ้า a < b เพื่อไม่ให้คำตอบติดลบ
        if a < b:
            a, b = b, a
        # คำนวนคำตอบที่ถูก
        correct = a - b
        # สรา้งข้อความโจทย์
        question = f"{a} - {b} = ?"
## โจทย์คูณ
    elif operation == "mul":
        # สุ่มเลข 1-12
        a, b = random.randint(1, 12), random.randint(1, 12)
        correct = a * b
        question = f"{a} × {b} = ?"
## โจทย์หาร
    elif operation == "div":
        # สุ่มตัวหาร 1–12
        b = random.randint(1, 12)
        # สุ่มคำตอบที่ถูกต้อง 1-12
        correct = random.randint(1, 12)
        # คำนวณหาตัวตั้ง คือการนำตัวหาร * คำตอบ
        a = b * correct
        question = f"{a} ÷ {b} = ?"
## สร้างตัวเลือกคำตอบ
    # เริ่มด้วยคำตอบที่ถูก ทำไมถึงใช้ set เพราะ set จะไม่เก็บค่าซ้ำจะไม่ทำให้มีตัวเลือกซ้ำ
    choices = set([correct])
    # สร้างตัวเลือกจนกว่าจะครบ 4
    while len(choices) < 4:
        # สุ่มค่าความต่างจากคำตอบจริง โดยสุ่มตัวเลข 1-9 เพื่อนำไปสร้างคำตอบหลอก
        delta = random.choice([1,2,3,4,5,6,7,8,9])
        # สุ่มเครื่องหมายว่าจะเป็น + หรือ -
        sign = random.choice([1, -1])
        # สร้างคำตอบหลอกโดยการนำ คำตอบที่ถูก + เครื่องหมาย +,- แล้วคูณด้วย ค่าความต่างจากคำตอบจริง
        cand = correct + sign * delta
        # ตรวจสอบไม่ให้มีค่าติดลบ ถ้าคำตอบไม่ติดลบให้เพิ่มใน ตัวเลือก
        if cand >= 0:
            choices.add(cand)
    # แปลงข้อมูลจาก set เป็น list เพื่อนำไปใช้กับ random.shuffle(choices)
    choices = list(choices)
    # สุ่มตำแหน่งของคำตอบ
    random.shuffle(choices)
    # ส่งค่ากลับ
    return question, correct, choices

# =========================
# เตรียมข้อสอบ 12 ข้อ
# =========================
# ตรวจสอบว่ามีคำถามน session(หน่วยความจำของ Streamlit) หรือยัง ถ้าไม่มีจะสร้างคำถามใหม่ ถ้ามีก็จะไม่สร้าง
if "questions" not in st.session_state:
    # สร้าง list ว่างเพื่อเก็บคำถามทั้งหมด
    st.session_state.questions = []
    # สร้าง list ของประเภทคำถามจะได้เป็นคำถามทักษะแต่ละทักษะจำนวน 3 ข้อ
    operations = ["add"]*3 + ["sub"]*3 + ["mul"]*3 + ["div"]*3
    # สุ่มลำดับคำถาม
    random.shuffle(operations)
    # สร้างคำถามแต่ละข้อ โดยจะจัดลูปตามประเภทคำถาม เช่น op = mul
    for op in operations:
        # เรียกฟังก์ชั่นสร้างโจทย์ generate_question() q คือ ข้อโจทย์ ans คือ คำตอบที่ถูก choices คือ ตัวเลือก
        q, ans, choices = generate_question(op)
        # เก็บคำถามลงใน session
        st.session_state.questions.append({
            # โครงสร้างคำถาม
            # ตย. mul
            # 7 * 8 = ?
            # 56
            # [54,56,60,58]
            "operation": op,
            "question": q,
            "answer": ans,
            "choices": choices
        })

# =========================
# แสดงข้อสอบ
# =========================
# สร้างตัวแปรเก็บคะแนนแต่ละทักษะ
scores = {"add":0, "sub":0, "mul":0, "div":0}
# หัวข้อแบบทดสอบ
st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
# แสดงหัวข้อเล็กๆใต้หัวข้อ
st.caption("โปรดตอบทุกข้อ แล้วกดปุ่ม 'ส่งคำตอบ'")
# กำหนดข้อความเริ่มต้น เพื่อให้ผู้ใช้เห็นว่ายังไม่ได้เลือกคำตอบ
PLACEHOLDER = "-- เลือกคำตอบ --"
## สร้างตัวแปรในการเก็บคำตอบของผู้ใช้
# ตรวจสอบว่ามี user_answers หรือยัง ถ้ามีไม่ต้องสร้างใหม่ ถ้าไม่มีจะสร้าง list เก็บคำตอบ ถ้าตรวจก็จะเกิดปัญหาคำตอบของผู้ใช้ถูกรีเซ็ตทุกครั้ง
if "user_answers" not in st.session_state:
    # สร้าง list เก็บคำตอบของผู้ใช้ list มีขนาดเท่ากับจำนวนข้อสอบ
    # ส่วนค่า PLACEHOLDER ใช้แทนยังไม่ได้ตอบ
    # len(st.session_state.questions) นับจำนวนข้อสอบ มีข้อสอบ 12 ข้อ จะได้ PLACEHOLDER * 12 ผลที่ได้จะมีเลือกคำตอบ 12 ช่อง
    st.session_state.user_answers = [PLACEHOLDER] * len(st.session_state.questions)
# วนลูปคำถามทุกข้อ i คือลำดับข้อ q ข้อมูลของคำถาม 
for i, q in enumerate(st.session_state.questions):
    ## สร้างตัวเลือกคำตอบจะมี 
    # PLACEHOLDER(-- เลือกคำตอบ --) 
    # [str(c) for c in q["choices"] ทำการแปลงตัวเลือกเป็นข้อความ ตย. q["choices"] = [23,20,18,17] --> ["23","20","18","17"] เพื่อจะนำไปใช้กับ st.radio()
    choices_with_placeholder = [PLACEHOLDER] + [str(c) for c in q["choices"]]
    # เป็นการสร้างตัวเลือก ○ ตัวเลือก 1 และเลือกได้ 1 คำตอบ ค่าที่เลือกจะถูกเก็บไว้ในตัวแปร selected
    selected = st.radio(
        # ข้อความคำถาม 
        # i+1 เพราะใน index จะเริ่มจาก 0 และดึงโจทย์มาจาก question
        # จะได้ ข้อ 1 : โจทย์
        f"ข้อ {i+1}: {q['question']}",
        # คือตัวเลือกตอบคำถาม
        # ○ -- เลือกคำตอบ --
        # ○ 23
        # ---
        choices_with_placeholder,
        # กำหนด key ให้แต้ละข้อ เช่น ข้อ 1 key q0 Streamlit ต้องใช้ key เพื่อจำค่าที่ผู้ใช้เลือกถ้าไม่ใส่ key อาจเกิด error
        key=f"q{i}"
    )
    # ค่าที่ผู้ใช้เลือกจะถูกเก็บไว้ในตัวแปร st.session_state.user_answers
    st.session_state.user_answers[i] = selected

# =========================
# ตรวจคำตอบ
# =========================
# สร้างปุ่มส่งคำตอบ
if st.button("ส่งคำตอบ"):
    # ตรวจสอบว่าทำครบทุกข้อรึยังถ้ายังก็จะแสดง กรุณาตอบให้ครบทุกข้อก่อนส่ง
    if any(ans == PLACEHOLDER for ans in st.session_state.user_answers):
        st.warning("กรุณาตอบให้ครบทุกข้อก่อนส่ง")
    # ตอบครบทุกข้อ
    else:
        # นับคะแนน
        # วนลูปคำถาม i ลำดับข้อ q โจทย์
        for i, q in enumerate(st.session_state.questions):
            # ดึงคำตอบของผู้ใช้ เช่นเลือก 20 จะได้ user_val = "20"
            user_val = st.session_state.user_answers[i]
            # แปลงเป็นตัวเลขจาก ตัวหนังสือ --> ตัวเลข
            try:
                user_val_num = int(user_val)
            except:
                user_val_num = None
            # ตรวจสอบว่าคำตอบถูกหรือไม่ โดยตรวจสอบว่าค่าที่ผู้ใช้เลือกตรงกับเฉลยหรือไม่
            if user_val_num == q["answer"]:
                #เพิ่มคะแนนแต่ละทักษะใน scores
                scores[q["operation"]] += 1
                ## หลังจากเพิ่มข้อมูลของคะแนนแต่ละทักษะเข้าไปใน scores แล้ว คะแนนจะถูกส่งไปยัง RandomForest เพื่อวิเคราะห์จุดอ่อนของผู้เรียนต่อ

        # แปลงเป็นข้อที่ตอบถูกเป็น % ในแต่ละทักษะ คะแนนดิบจะอยู่ที่ 0-3
        # (จำนวนข้อที่ถูก/จำนวนข้อทั้งหมด) * 100
        add_score = round((scores["add"]/3)*100, 2)
        sub_score = round((scores["sub"]/3)*100, 2)
        mul_score = round((scores["mul"]/3)*100, 2)
        div_score = round((scores["div"]/3)*100, 2)
        #แสดงคำตอบที่ถูกแบบ %
        st.subheader("📊 ผลคะแนน")
        st.write(f"➕ การบวก: {add_score}")
        st.write(f"➖ การลบ: {sub_score}")
        st.write(f"✖ การคูณ: {mul_score}")
        st.write(f"➗ การหาร: {div_score}")

        # -------------------------
        # วิเคราะห์จากคะแนนจริง (หลัก)
        # -------------------------
        # เก็บคะแนนแต่ละทักษะแบบ %
        skill_scores = {
            "add": add_score,
            "sub": sub_score,
            "mul": mul_score,
            "div": div_score
        }
        # สร้างชื่อและลิ้งค์วิดีโอไว้ใน friendly
        friendly = {
            "add": ("การบวก", "https://www.youtube.com/watch?v=c5eS7nRsE_Q"),
            "sub": ("การลบ", "https://www.youtube.com/watch?v=vT_VBLlvdn8"),
            "mul": ("การคูณ", "https://www.youtube.com/watch?v=73obrcsERe8"),
            "div": ("การหาร", "https://www.youtube.com/watch?v=9D1JW8rYqeA")
        }

        # เกณฑ์ผ่านถ้าทุกทักษะ >= 60 ถือว่าพื้นฐานดี
        pass_threshold = 60

        # หาคะแนนที่ต่ำที่สุด และหาทักษะที่ต่ำที่สุดทั้งหมด
        # min_score ค่าต่ำสุดของคะแนนทั้ง 4 ทักษะ
        min_score = min(skill_scores.values())
        # รายชื่อทักษะที่ได้คะแนนเท่ากับ min_score ทั้งหมด ถ้าเท่ากับ min จะแสดงออกมาด้วย
        weakest = [k for k, v in skill_scores.items() if v == min_score]
        
        st.subheader("🔎 วิเคราะห์จากคะแนนจริง (แนะนำหลัก)")
        # ถ้าทุกทักษะผ่านหมด แสดงว่าไม่มีจุดอ่อนชัดเจน (ถ้าคะแนนทั้ง 4 >= 60)
        if all(score >= pass_threshold for score in skill_scores.values()):
            st.success("พื้นฐานคณิตศาสตร์อยู่ในระดับดี ไม่มีจุดอ่อนที่ชัดเจน 🎉")
        # ถ้ามีทักษะต่ำกว่าเกณฑ์ ระบบจะแนะนำหัวข้อที่ควรพัฒนา
        else:
            # บอกคะแนนต่ำสุด
            st.write(f"คะแนนต่ำสุดคือ {min_score} — หัวข้อที่ควรพัฒนา:")
            # วนลูปทุกทักษะใน weakest แล้วแสดงชื่อและวิดีโอ
            for w in weakest:
                name, vid = friendly[w]
                st.write(f"- {name} (คะแนน {skill_scores[w]} / 100)")
                st.write(f"→ คำแนะนำ: ฝึก{name}เพิ่มเติม")
                st.video(vid)

        # -------------------------
        # AI Prediction (เสริม)
        # -------------------------
        st.subheader("🤖 ผลการวิเคราะห์จาก AI (เสริม)")

        # เตรียมข้อมูลเข้าโมเดล โดยมีคะแนนแต่ละทักษะ %
        X_input = [[add_score, sub_score, mul_score, div_score]]
        # ทำการทำนาย class ที่โมเดลคิดว่ารูปแบบคะแนนคล้ายกับ class ไหนที่สุด
        # [0] เพื่อเวลาดึงค่าออกมาแบบไม่ใส่ [0] จะได้ ["class"] จึงใส่เพื่อแปลงเป็น "class"
        result = model.predict(X_input)[0]
        # เอาไว้ดูความน่าจะเป็นของทุก class 
        proba = model.predict_proba(X_input)[0]
        # ดึงรายชื่อ class ทั้งหมดที่โมเดลเคยเรียกจาก data จะได้ classes = ["weak_add", "weak_sub", "weak_mul", "weak_div"]
        classes = list(model.classes_)
        # จับคู่ชื่อ class กับความน่าจะเป็น
        # จะได้ ตย. "weak_add": 0.10
        proba_map = {c: float(p) for c, p in zip(classes, proba)}
        # แสดงผลที่ทำนายบนหน้าเว็ป
        st.info(f"โมเดลทำนายจุดที่ควรพัฒนา: {result}")

        # แปลงค่าผลลัพธ์ที่ Ai ทำนายในโมเดล ให้กลายเป็นชื่อทักษะที่ใช้ในระบบ "weak_add": "add"
        model_map = {
            "weak_add": "add",
            "weak_sub": "sub",
            "weak_mul": "mul",
            "weak_div": "div",
        }
        #นำข้อมูลที่แปลงค่าไปใส่ในตัวแปร predicted_skill
        predicted_skill = model_map.get(result)

        ## เงื่อนไข ถ้าโมเดลทายไม่ตรงกับ “จุดอ่อนจากคะแนนจริง”
        # ให้แจ้งว่าใช้คะแนนจริงเป็นหลัก และโมเดลเป็นเสริม
        # ตรวจสอบว่า Ai ทำนายไม่ตรงกับจุดอ่อนคะแนนจริงหรือไม่ ถ้าไม่ตรงก็จะแสดงข้อความ
        # หลักการทำงาน คือ จะนำ class ที่ระบบคะแนนจริงมาเที่ยบกับ class ที่ Ai ทำนายออกมา
        if predicted_skill and predicted_skill not in weakest and not all(score >= pass_threshold for score in skill_scores.values()):
            st.warning(
                "หมายเหตุ: กรณีนี้คะแนนจริงชี้ชัดว่าควรพัฒนาหัวข้อที่คะแนนต่ำสุดก่อน "
                "ส่วนผลจากโมเดลเป็นการทำนายจากรูปแบบในชุดข้อมูล จึงอาจไม่ตรงกับคะแนนจริงได้"
            )

        # แสดงความน่าจะเป็น 
        with st.expander("ดูความน่าจะเป็นของโมเดล"):
            # เรียงจากมากไปน้อยจากคำสั่ง sorted()
            sorted_items = sorted(proba_map.items(), key=lambda x: x[1], reverse=True)
            st.write("ความน่าจะเป็นที่โมเดลให้แต่ละคลาส:")
            # วนลูปแสดงผล c ชื่อ class p ความน่าจะเป็น 
            for c, p in sorted_items:
                #แสดงผลลัพธ์เป็น %
                st.write(f"- {c}: {round(p*100, 2)}%")

# =========================
# ปุ่มเริ่มใหม่
# =========================
if st.button("🔄 เริ่มใหม่"):
    keys_to_clear = ["questions", "user_answers"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

    try:
        st.experimental_rerun()
    except Exception:
        st.stop()
