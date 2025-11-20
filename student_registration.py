import streamlit as st
import sqlite3
import io
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
import face_recognition
from datetime import datetime

# Set TensorFlow logging level to reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize face detector with improved parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize database
def init_db():
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        # Create classes table
        c.execute('''CREATE TABLE IF NOT EXISTS classes
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     class_name TEXT UNIQUE NOT NULL)''')
        
        # Create students table with class reference
        c.execute('''CREATE TABLE IF NOT EXISTS students
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     class_id INTEGER NOT NULL,
                     name TEXT NOT NULL,
                     age INTEGER NOT NULL,
                     email TEXT UNIQUE NOT NULL,
                     image BLOB,
                     face_encoding BLOB,
                     FOREIGN KEY (class_id) REFERENCES classes(id))''')
        
        # Create attendance table
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     student_id INTEGER NOT NULL,
                     class_id INTEGER NOT NULL,
                     date TEXT NOT NULL,
                     period TEXT NOT NULL,
                     status TEXT NOT NULL,
                     FOREIGN KEY (student_id) REFERENCES students(id),
                     FOREIGN KEY (class_id) REFERENCES classes(id))''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {str(e)}")

# Initialize database on startup
init_db()

def extract_face_features(image):
    """提取单张人脸特征（用于学生注册）"""
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return None
    
    # Get face encodings for the faces
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return None
    
    # Return the first face encoding
    return face_encodings[0]

def extract_all_face_features(image):
    """提取照片中所有人脸特征(用于考勤打卡)"""
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 尝试使用 HOG 模型检测人脸（更快但可能检测较少）
        face_locations_hog = face_recognition.face_locations(rgb_image, model='hog')
        
        # 如果 HOG 检测不到，尝试使用 CNN 模型（更准确但较慢）
        if not face_locations_hog:
            st.info("🔍 HOG 模型未检测到人脸，正在使用 CNN 模型重新检测...")
            try:
                face_locations_cnn = face_recognition.face_locations(rgb_image, model='cnn')
                face_locations = face_locations_cnn
            except:
                face_locations = []
        else:
            face_locations = face_locations_hog
        
        # 如果还是检测不到，尝试调整图像
        if not face_locations:
            st.warning("⚠️ 未检测到人脸，尝试增强图像...")
            # 增强对比度
            enhanced = cv2.convertScaleAbs(rgb_image, alpha=1.2, beta=30)
            face_locations = face_recognition.face_locations(enhanced, model='hog', number_of_times_to_upsample=2)
            if face_locations:
                st.success(f"✅ 图像增强后检测到 {len(face_locations)} 张人脸")
                rgb_image = enhanced
        
        if not face_locations:
            st.error("❌ 尝试多种方法后仍未检测到人脸")
            st.info("💡 建议：\n1. 确保照片光线充足\n2. 人脸清晰可见且正面朝向\n3. 照片分辨率足够高\n4. 尝试裁剪照片使人脸更大")
            return [], []
        
        st.success(f"✅ 检测到 {len(face_locations)} 张人脸，正在提取特征...")
        
        # Get face encodings for all faces
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            st.error("检测到人脸位置，但无法提取人脸特征")
            return [], []
        
        return face_encodings, face_locations
    except Exception as e:
        st.error(f"提取人脸特征时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return [], []

def compare_faces(encoding1, encoding2):
    if encoding1 is None or encoding2 is None:
        return 0
    
    try:
        # Convert stored encoding back to numpy array
        encoding2 = np.frombuffer(encoding2, dtype=np.float64)
        
        # Calculate face distance (lower is better)
        face_distance = face_recognition.face_distance([encoding2], encoding1)[0]
        
        # Convert distance to similarity score (1 - distance)
        similarity = 1 - face_distance
        
        return max(0, min(1, similarity))
    except Exception as e:
        st.error(f"Error comparing faces: {str(e)}")
        return 0

# Function to create a database connection
def create_connection():
    try:
        conn = sqlite3.connect("students.db")
        return conn
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to verify database structure
def verify_database():
    conn = create_connection()
    if conn is None:
        return False
    
    cursor = conn.cursor()
    try:
        # Check if sqlite_master table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            st.info("Database is empty. It will be initialized when you register the first student.")
            return True
            
        return True
    except sqlite3.Error as e:
        st.error(f"Database verification error: {e}")
        return False
    finally:
        conn.close()

# Function to create a class table if it doesn't exist
def create_class_table(class_name):
    if not class_name:
        st.error("Please enter a valid class name")
        return False
    
    # Sanitize class name to prevent SQL injection
    class_name = ''.join(c for c in class_name if c.isalnum() or c == '_')
    
    conn = create_connection()
    if conn is None:
        return False
        
    cursor = conn.cursor()
    try:
        # Create table with correct schema
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS {class_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT UNIQUE NOT NULL,
            image BLOB NOT NULL,
            face_encoding BLOB NOT NULL
        )''')
        
        # Create attendance table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            date TEXT NOT NULL,
            period TEXT NOT NULL,
            status TEXT NOT NULL
        )''')
        
        conn.commit()
        st.success(f"Class table '{class_name}' created successfully!")
        return True
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return False
    finally:
        conn.close()

# Function to convert binary data to an image
def convert_to_image(data):
    return Image.open(io.BytesIO(data))

def create_class(class_name):
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("INSERT INTO classes (class_name) VALUES (?)", (class_name,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error("Class already exists!")
        return False
    except Exception as e:
        st.error(f"Error creating class: {str(e)}")
        return False

def get_all_classes():
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("SELECT id, class_name FROM classes")
        classes = c.fetchall()
        conn.close()
        return classes
    except Exception as e:
        st.error(f"Error fetching classes: {str(e)}")
        return []

def register_student(class_id, name, photo, age, email):
    try:
        # Convert photo to numpy array
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Extract face encoding
        face_encoding = extract_face_features(image)
        if face_encoding is None:
            st.error("No face detected in the image. Please try again with a clearer photo.")
            return False
        
        # Store in database
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        
        # Insert the student data
        c.execute("INSERT INTO students (class_id, name, age, email, image, face_encoding) VALUES (?, ?, ?, ?, ?, ?)",
                 (class_id, name, age, email, file_bytes, face_encoding.tobytes()))
        conn.commit()
        conn.close()
        
        st.success(f"Student {name} registered successfully!")
        return True
    except Exception as e:
        st.error(f"Error registering student: {str(e)}")
        return False

def list_students(class_id):
    """列出班级中所有已注册的学生"""
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.name, s.age, s.email, s.image
            FROM students s
            WHERE s.class_id = ?
        """, (class_id,))
        
        students = cursor.fetchall()
        
        if not students:
            st.info("该班级暂无注册学生。")
            return None
        
        # 显示学生信息和照片
        st.write(f"### 共有 {len(students)} 名学生")
        
        # 使用列布局显示学生卡片
        for idx, (student_id, name, age, email, image_data) in enumerate(students):
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # 显示学生照片
                    if image_data:
                        try:
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, width=150, caption=f"学生照片")
                        except:
                            st.warning("照片加载失败")
                    else:
                        st.info("无照片")
                
                with col2:
                    # 显示学生信息
                    st.markdown(f"""
                    **学生 ID:** {student_id}  
                    **姓名:** {name}  
                    **年龄:** {age}  
                    **邮箱:** {email}
                    """)
                
                st.divider()
        
        return True
        
    except Exception as e:
        st.error(f"查询学生列表时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
    finally:
        conn.close()

def update_student(student_id, class_id, name=None, age=None, email=None, photo=None):
    """修改学生信息"""
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        # 验证学生是否存在
        cursor.execute("""
            SELECT id FROM students 
            WHERE id = ? AND class_id = ?
        """, (student_id, class_id))
        
        if not cursor.fetchone():
            st.error(f"学生 ID {student_id} 不存在或不属于该班级")
            conn.close()
            return False
        
        # 构建更新语句
        update_fields = []
        update_values = []
        
        if name is not None:
            update_fields.append("name = ?")
            update_values.append(name)
        
        if age is not None:
            update_fields.append("age = ?")
            update_values.append(age)
        
        if email is not None:
            update_fields.append("email = ?")
            update_values.append(email)
        
        if photo is not None:
            # 处理新照片
            file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # 提取人脸特征
            face_encoding = extract_face_features(image)
            if face_encoding is None:
                st.error("新照片中未检测到人脸，请使用清晰的照片")
                conn.close()
                return False
            
            update_fields.append("image = ?")
            update_values.append(file_bytes)
            update_fields.append("face_encoding = ?")
            update_values.append(face_encoding.tobytes())
        
        if not update_fields:
            st.warning("没有需要更新的字段")
            conn.close()
            return False
        
        # 执行更新
        update_values.extend([student_id, class_id])
        sql = f"UPDATE students SET {', '.join(update_fields)} WHERE id = ? AND class_id = ?"
        cursor.execute(sql, update_values)
        
        conn.commit()
        conn.close()
        
        st.success(f"✅ 学生信息更新成功！")
        return True
        
    except sqlite3.IntegrityError as e:
        st.error(f"邮箱已被其他学生使用！")
        return False
    except Exception as e:
        st.error(f"更新学生信息时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def delete_student(student_id, class_id):
    """删除学生"""
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        # 验证学生是否存在
        cursor.execute("""
            SELECT name FROM students 
            WHERE id = ? AND class_id = ?
        """, (student_id, class_id))
        
        result = cursor.fetchone()
        if not result:
            st.error(f"学生 ID {student_id} 不存在或不属于该班级")
            conn.close()
            return False
        
        student_name = result[0]
        
        # 删除考勤记录
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        
        # 删除学生
        cursor.execute("DELETE FROM students WHERE id = ? AND class_id = ?", (student_id, class_id))
        
        conn.commit()
        conn.close()
        
        st.success(f"✅ 学生 {student_name} (ID: {student_id}) 已被删除！")
        return True
        
    except Exception as e:
        st.error(f"删除学生时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def fetch_student(student_id, class_id):
    """Fetch student details by ID"""
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.name, s.age, s.email, s.face_encoding 
            FROM students s
            WHERE s.id = ? AND s.class_id = ?
        """, (student_id, class_id))
        
        student = cursor.fetchone()
        
        if not student:
            st.error("Student not found in this class.")
            return None
        
        # Get attendance records
        cursor.execute("""
            SELECT date, period, status 
            FROM attendance 
            WHERE student_id = ? AND class_id = ?
            ORDER BY date DESC, period
        """, (student_id, class_id))
        
        attendance_records = cursor.fetchall()
        
        return student, attendance_records

    except Exception as e:
        st.error(f"Error fetching student: {str(e)}")
        return None
    finally:
        conn.close()

def mark_attendance(class_id, photos, period):
    """从多张集合照片中识别所有学生并标记考勤"""
    try:
        # Get all students from database for this class
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("SELECT id, name, face_encoding FROM students WHERE class_id = ?", (class_id,))
        students = c.fetchall()
        
        if not students:
            st.error("该班级没有已注册的学生。")
            conn.close()
            return
        
        st.info(f"👥 班级共有 {len(students)} 名注册学生")
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 存储识别结果（使用 dict 避免重复）
        recognized_students = {}  # {student_id: (name, max_similarity)}
        total_faces_detected = 0
        total_faces_unrecognized = 0
        
        # 处理每张照片
        st.write(f"### 📸 处理 {len(photos)} 张照片")
        
        for photo_idx, photo in enumerate(photos, 1):
            st.write(f"---")
            st.write(f"#### 🖼️ 照片 {photo_idx}/{len(photos)}: {photo.name}")
            
            # Convert photo to numpy array
            file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error(f"✖️ 无法解码照片 {photo.name}，跳过")
                continue
            
            # 显示图片信息
            height, width = image.shape[:2]
            st.caption(f"📊 图片尺寸: {width}x{height} 像素")
            
            # 如果图片太大，进行缩放以加快处理速度
            max_dimension = 1600
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                st.caption(f"🔄 已缩放至 {new_width}x{new_height}")
            
            # 提取照片中所有人脸的编码
            with st.spinner(f'🔍 正在检测照片 {photo_idx} 中的人脸...'):
                face_encodings, face_locations = extract_all_face_features(image)
            
            if not face_encodings:
                st.warning(f"⚠️ 照片 {photo.name} 中未检测到人脸")
                continue
            
            st.success(f"✅ 检测到 {len(face_encodings)} 张人脸")
            total_faces_detected += len(face_encodings)
            
            photo_unrecognized = 0
            photo_recognized = []
            
            # 为照片中的每张人脸找到最佳匹配学生
            for face_idx, face_encoding in enumerate(face_encodings):
                best_match = None
                best_similarity = 0
                
                # 与所有学生进行比对
                for student_id, name, stored_encoding in students:
                    try:
                        stored_encoding_array = np.frombuffer(stored_encoding, dtype=np.float64)
                        face_distance = face_recognition.face_distance([stored_encoding_array], face_encoding)[0]
                        similarity = 1 - face_distance
                        
                        # 找到相似度最高的学生
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (student_id, name, similarity)
                    except Exception as e:
                        continue
                
                # 降低阈值到 0.4 以提高识别率
                threshold = 0.4
                if best_match and best_similarity >= threshold:
                    student_id = best_match[0]
                    name = best_match[1]
                    
                    # 更新识别结果（保留最高相似度）
                    if student_id not in recognized_students:
                        recognized_students[student_id] = (name, best_similarity)
                        photo_recognized.append(f"{name} ({best_similarity*100:.1f}%)")
                    else:
                        # 如果这次识别的相似度更高，更新
                        if best_similarity > recognized_students[student_id][1]:
                            recognized_students[student_id] = (name, best_similarity)
                        photo_recognized.append(f"{name} (重复)")
                else:
                    photo_unrecognized += 1
                    total_faces_unrecognized += 1
            
            # 显示该照片的识别结果
            if photo_recognized:
                st.info(f"👤 该照片识别到: {', '.join(photo_recognized)}")
            if photo_unrecognized > 0:
                st.warning(f"⚠️ 该照片有 {photo_unrecognized} 张人脸未识别")
        
        # 标记考勤
        st.write("---")
        st.write("### 📝 正在保存考勤记录...")
        
        present_students = []
        for student_id, (name, similarity) in recognized_students.items():
            # Check if attendance already exists
            c.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND class_id = ? AND date = ? AND period = ?
            """, (student_id, class_id, current_date, period))
            
            existing = c.fetchone()
            if not existing:
                c.execute("""
                    INSERT INTO attendance (student_id, class_id, date, period, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (student_id, class_id, current_date, period, 'Present'))
                present_students.append((student_id, name, similarity))
            else:
                st.info(f"ℹ️ 学生 {name} (ID: {student_id}) 在本节课已有考勤记录")
        
        conn.commit()
        
        # 获取所有学生列表，计算缺勤
        c.execute("SELECT id, name FROM students WHERE class_id = ?", (class_id,))
        all_students = c.fetchall()
        conn.close()
        
        # 计算缺勤学生
        present_ids = list(recognized_students.keys())
        absent_students = [(sid, sname) for sid, sname in all_students if sid not in present_ids]
        
        # 显示汇总统计
        st.write("---")
        st.write("### 📊 考勤汇总统计")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("处理照片数", len(photos))
        with col2:
            st.metric("检测人脸数", total_faces_detected)
        with col3:
            st.metric("识别成功", len(recognized_students), delta="出勤")
        with col4:
            st.metric("缺勤人数", len(absent_students))
        
        # 显示出勤学生详情
        if present_students:
            st.write("### ✅ 出勤学生")
            attendance_data = []
            for student_id, name, similarity in present_students:
                attendance_data.append({
                    "学生ID": student_id,
                    "姓名": name,
                    "最高匹配度": f"{similarity*100:.1f}%"
                })
            st.dataframe(pd.DataFrame(attendance_data), use_container_width=True)
        
        # 显示缺勤学生
        if absent_students:
            st.write("### ❌ 缺勤学生")
            absent_data = []
            for student_id, name in absent_students:
                absent_data.append({
                    "学生ID": student_id,
                    "姓名": name
                })
            st.dataframe(pd.DataFrame(absent_data), use_container_width=True)
        
        # 总结信息
        if total_faces_unrecognized > 0:
            st.warning(f"⚠️ 总共有 {total_faces_unrecognized} 张人脸未能识别，可能原因：\n" + 
                      "1. 该学生未在系统中注册\n" +
                      "2. 注册照片与现场照片差异较大\n" +
                      "3. 照片质量或角度问题\n" +
                      "4. 人脸被遮挡或不清晰")
        
        st.success(f"✅ 考勤标记完成！共处理 {len(photos)} 张照片，{len(recognized_students)} 名学生出勤。")
        
    except Exception as e:
        st.error(f"标记考勤时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def view_attendance(class_id):
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get all students in the class
        cursor.execute("""
            SELECT s.id, s.name 
            FROM students s 
            WHERE s.class_id = ?
        """, (class_id,))
        students = cursor.fetchall()
        
        if not students:
            st.info("该班级暂无注册学生。")
            return
        
        # Get attendance for today
        cursor.execute("""
            SELECT s.id, s.name, a.period, a.status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id 
                AND a.class_id = s.class_id 
                AND a.date = ?
            WHERE s.class_id = ?
            ORDER BY s.name, a.period
        """, (current_date, class_id))
        
        attendance_records = cursor.fetchall()
        
        if not attendance_records:
            st.warning(f"📅 {current_date} 暂无考勤记录")
            st.info(f"班级共有 {len(students)} 名学生，请先进行考勤打卡。")
            return
        
        # Create a DataFrame
        df = pd.DataFrame(attendance_records, columns=['ID', 'Name', 'Period', 'Status'])
        
        # 过滤掉没有考勤记录的行（Period 为 None）
        df_filtered = df[df['Period'].notna()].copy()
        
        if df_filtered.empty:
            st.warning(f"📅 {current_date} 暂无考勤记录")
            st.info(f"班级共有 {len(students)} 名学生，请先进行考勤打卡。")
            return
        
        # Pivot the data to show periods as columns
        pivot_df = df_filtered.pivot_table(
            index=['ID', 'Name'], 
            columns='Period', 
            values='Status',
            aggfunc='first'
        )
        pivot_df = pivot_df.reset_index()
        
        # 填充空值为 "缺勤"
        pivot_df = pivot_df.fillna("-")
        
        # Calculate attendance statistics
        total_students = len(students)
        # 统计有至少一次出勤记录的学生数
        present_students = len(df_filtered[df_filtered['Status'] == 'Present']['ID'].unique())
        absent_students = total_students - present_students
        
        # 统计各节课的出勤情况
        periods_stats = df_filtered[df_filtered['Status'] == 'Present'].groupby('Period').size().to_dict()
        
        st.write(f"### 📊 {current_date} 考勤统计")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总学生数", total_students)
        with col2:
            st.metric("已出勤", present_students, delta="人")
        with col3:
            st.metric("未出勤", absent_students)
        
        # 显示各节课的出勤人数
        if periods_stats:
            st.write("#### 📚 各节课出勤情况")
            
            # 将节次按数字排序
            sorted_periods = sorted(periods_stats.items(), key=lambda x: int(x[0]))
            
            # 如果节次较多（超过5个），分多行显示
            if len(sorted_periods) > 5:
                # 分成两行显示
                row1_periods = sorted_periods[:5]
                row2_periods = sorted_periods[5:]
                
                # 第一行
                period_cols_row1 = st.columns(5)
                for idx, (period, count) in enumerate(row1_periods):
                    with period_cols_row1[idx]:
                        st.metric(f"第{period}节", f"{count}人")
                
                # 第二行
                if row2_periods:
                    period_cols_row2 = st.columns(len(row2_periods))
                    for idx, (period, count) in enumerate(row2_periods):
                        with period_cols_row2[idx]:
                            st.metric(f"第{period}节", f"{count}人")
            else:
                # 节次较少，一行显示
                period_cols = st.columns(len(sorted_periods))
                for idx, (period, count) in enumerate(sorted_periods):
                    with period_cols[idx]:
                        st.metric(f"第{period}节", f"{count}人")
        
        st.write("### 📋 详细考勤表")
        st.info("💡 提示:表格支持横向滚动查看所有节次")
        
        # 使用 Streamlit 的 column_config 来优化表格显示
        # 设置表格宽度和列宽配置
        column_config = {
            "ID": st.column_config.NumberColumn(
                "学生ID",
                width="small",
            ),
            "Name": st.column_config.TextColumn(
                "姓名",
                width="medium",
            ),
        }
        
        # 为每个节次列添加配置
        for col in pivot_df.columns:
            if col not in ['ID', 'Name']:
                column_config[col] = st.column_config.TextColumn(
                    f"第{col}节",
                    width="small",
                )
        
        # 显示表格,使用 use_container_width=True 让表格占满容器宽度
        st.dataframe(
            pivot_df, 
            use_container_width=True,
            column_config=column_config,
            hide_index=True,
            height=400  # 设置表格高度,支持垂直滚动
        )
        
    except Exception as e:
        st.error(f"查看考勤时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        conn.close()

# Streamlit UI
st.title("Smart Attendance System")

# Verify database at startup
if not verify_database():
    st.error("Database verification failed. Please try deleting and recreating the database.")

# Get all classes
classes = get_all_classes()
class_options = {f"{c[1]}": c[0] for c in classes}

# Add class selection at the top
if not class_options:
    st.warning("No classes found. Please create a class first.")
    new_class = st.text_input("Enter new class name")
    if st.button("Create Class") and new_class:
        if create_class(new_class):
            st.success(f"Class {new_class} created successfully!")
            st.rerun()
else:
    selected_class = st.selectbox("Select Class", options=list(class_options.keys()))
    class_id = class_options[selected_class]

    menu = ["Register Student", "List Students", "Update Student", "Delete Student", "Fetch Student Details", "Mark Attendance", "View Attendance", "Delete Database"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Register Student":
        st.subheader("Register a New Student")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=100)
        email = st.text_input("Email")
        photo = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

        if st.button("Register") and photo is not None:
            register_student(class_id, name, photo, age, email)

    elif choice == "Update Student":
        st.subheader("修改学生信息")
        
        # 显示学生列表供选择
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name FROM students WHERE class_id = ?
        """, (class_id,))
        students = cursor.fetchall()
        conn.close()
        
        if not students:
            st.info("该班级暂无注册学生")
        else:
            # 创建学生选择字典
            student_options = {f"{s[1]} (ID: {s[0]})": s[0] for s in students}
            
            selected_student = st.selectbox("选择要修改的学生", options=list(student_options.keys()))
            student_id = student_options[selected_student]
            
            st.write("---")
            st.write("ℹ️ 请输入需要修改的信息（不修改的字段请留空）")
            
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("新姓名（留空不修改）")
                new_age = st.number_input("新年龄（0为不修改）", min_value=0, max_value=100, value=0)
            with col2:
                new_email = st.text_input("新邮箱（留空不修改）")
                new_photo = st.file_uploader("新照片（不上传为不修改）", type=["jpg", "jpeg", "png"], key="update_photo")
            
            if st.button("保存修改"):
                # 准备更新参数
                update_name = new_name if new_name else None
                update_age = new_age if new_age > 0 else None
                update_email = new_email if new_email else None
                update_photo = new_photo
                
                if update_student(student_id, class_id, update_name, update_age, update_email, update_photo):
                    st.balloons()
                    st.info("🔄 请刷新页面查看更新后的信息")
    
    elif choice == "Delete Student":
        st.subheader("删除学生")
        
        # 显示学生列表供选择
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, age, email FROM students WHERE class_id = ?
        """, (class_id,))
        students = cursor.fetchall()
        conn.close()
        
        if not students:
            st.info("该班级暂无注册学生")
        else:
            # 创建学生选择字典
            student_options = {f"{s[1]} (ID: {s[0]})": s[0] for s in students}
            
            selected_student = st.selectbox("选择要删除的学生", options=list(student_options.keys()))
            student_id = student_options[selected_student]
            
            # 显示学生详细信息
            student_info = [s for s in students if s[0] == student_id][0]
            st.write("---")
            st.write("### 学生信息")
            st.write(f"**ID:** {student_info[0]}")
            st.write(f"**姓名:** {student_info[1]}")
            st.write(f"**年龄:** {student_info[2]}")
            st.write(f"**邮箱:** {student_info[3]}")
            
            st.warning("⚠️ 警告：删除学生后将同时删除该学生的所有考勤记录，此操作不可恢复！")
            
            confirm = st.checkbox(f"我确认要删除学生 {student_info[1]}")
            
            if confirm and st.button("确认删除", type="primary"):
                if delete_student(student_id, class_id):
                    st.info("🔄 请刷新页面")

    elif choice == "List Students":
        st.subheader("学生列表")
        list_students(class_id)

    elif choice == "Fetch Student Details":
        st.subheader("Fetch Student Details")
        student_id = st.number_input("Enter Student ID", min_value=1)

        if st.button("Fetch"):
            result = fetch_student(student_id, class_id)
            if result:
                student, attendance_records = result
                st.write("### Student Information")
                st.write(f"ID: {student[0]}")
                st.write(f"Name: {student[1]}")
                st.write(f"Age: {student[2]}")
                st.write(f"Email: {student[3]}")
                
                if attendance_records:
                    st.write("### Attendance History")
                    attendance_df = pd.DataFrame(attendance_records, columns=['Date', 'Period', 'Status'])
                    st.dataframe(attendance_df)
                else:
                    st.info("No attendance records found for this student.")

    elif choice == "Mark Attendance":
        st.subheader("考勤打卡")
        period = st.selectbox("选择节次", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
        photos = st.file_uploader("上传课堂照片（可多选）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if st.button("开始考勤打卡") and photos:
            if len(photos) == 0:
                st.error("请至少上传一张照片")
            else:
                st.info(f"📸 共上传了 {len(photos)} 张照片")
                mark_attendance(class_id, photos, period)

    elif choice == "View Attendance":
        st.subheader("View Attendance")
        view_attendance(class_id)
        
    elif choice == "Delete Database":
        st.subheader("Delete Database")
        st.warning("⚠️ Warning: This action will delete all data including classes, students, and attendance records. This cannot be undone!")
        if st.button("Delete Database", type="primary"):
            if os.path.exists("students.db"):
                os.remove("students.db")
                st.success("Database deleted successfully.")
                init_db()  # Reinitialize the database
                st.rerun()
            else:
                st.error("Database does not exist.")
