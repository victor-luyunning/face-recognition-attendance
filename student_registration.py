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
    """List all registered students in a class"""
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.name, s.age, s.email 
            FROM students s
            WHERE s.class_id = ?
        """, (class_id,))
        
        students = cursor.fetchall()
        
        if not students:
            st.info("No students registered in this class yet.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(students, columns=['ID', 'Name', 'Age', 'Email'])
        return df
        
    except Exception as e:
        st.error(f"Error listing students: {str(e)}")
        return None
    finally:
        conn.close()

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

def mark_attendance(class_id, photo, period):
    try:
        # Convert photo to numpy array
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to decode the image. Please try again with a different photo.")
            return
        
        # Extract face encoding from the group photo
        face_encoding = extract_face_features(image)
        if face_encoding is None:
            st.error("No face detected in the image. Please try again with a clearer photo.")
            return
        
        # Get all students from database for this class
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("SELECT id, name, face_encoding FROM students WHERE class_id = ?", (class_id,))
        students = c.fetchall()
        conn.close()
        
        if not students:
            st.error("No students registered in this class.")
            return
        
        # Compare with each student's encoding
        matches = []
        for student_id, name, stored_encoding in students:
            try:
                similarity = compare_faces(face_encoding, stored_encoding)
                matches.append((student_id, name, similarity))
            except Exception as e:
                st.warning(f"Error comparing with student {name}: {str(e)}")
                continue
        
        if not matches:
            st.error("Failed to compare faces with any registered students.")
            return
        
        # Sort by similarity
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize lists for present and absent students
        present_students = []
        absent_students = []
        
        # Process each student
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        
        for student_id, name, similarity in matches:
            if similarity >= 0.5:  # Lowered threshold for better recognition
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
                    present_students.append(name)
            else:
                absent_students.append(name)
        
        conn.commit()
        conn.close()
        
        # Display attendance summary
        st.write("### Attendance Summary")
        st.write(f"**Present Students ({len(present_students)}):**")
        for student in present_students:
            st.write(f"- {student}")
        
        st.write(f"**Absent Students ({len(absent_students)}):**")
        for student in absent_students:
            st.write(f"- {student}")
        
    except Exception as e:
        st.error(f"Error marking attendance: {str(e)}")

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
            st.info("No students registered in this class.")
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
        
        # Create a DataFrame
        df = pd.DataFrame(attendance_records, columns=['ID', 'Name', 'Period', 'Status'])
        
        # Pivot the data to show periods as columns
        pivot_df = df.pivot(index=['ID', 'Name'], columns='Period', values='Status')
        pivot_df = pivot_df.reset_index()
        
        # Calculate attendance statistics
        total_students = len(students)
        present_students = len(df[df['Status'] == 'Present'].groupby('ID').count())
        absent_students = total_students - present_students
        
        st.write(f"### Attendance Statistics for {current_date}")
        st.write(f"Total Students: {total_students}")
        st.write(f"Present: {present_students}")
        st.write(f"Absent: {absent_students}")
        
        st.write("### Attendance Details")
        st.dataframe(pivot_df)
        
    except Exception as e:
        st.error(f"Error viewing attendance: {str(e)}")
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

    menu = ["Register Student", "List Students", "Fetch Student Details", "Mark Attendance", "View Attendance", "Delete Database"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Register Student":
        st.subheader("Register a New Student")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=100)
        email = st.text_input("Email")
        photo = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

        if st.button("Register") and photo is not None:
            register_student(class_id, name, photo, age, email)

    elif choice == "List Students":
        st.subheader("List All Students")
        if st.button("Show Students"):
            students_df = list_students(class_id)
            if students_df is not None:
                st.dataframe(students_df)

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
        st.subheader("Mark Attendance")
        period = st.selectbox("Select Period", ["1", "2", "3", "4", "5", "6", "7", "8"])
        photo = st.file_uploader("Upload Group Photo", type=["jpg", "jpeg", "png"])

        if st.button("Mark Attendance") and photo is not None:
            mark_attendance(class_id, photo, period)

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
