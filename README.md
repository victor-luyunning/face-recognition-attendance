# SmartAttend - Facial Recognition Attendance System

A modern attendance management system that uses facial recognition to automate the attendance process. Built with Streamlit and Python, this system provides an intuitive interface for managing student attendance in educational institutions.

## Features

- **Class Management**
  - Create and manage multiple classes
  - Organize students by class
  - Easy class selection for all operations

- **Student Registration**
  - Register students with their details
  - Upload student photos
  - Automatic face encoding for recognition

- **Attendance Management**
  - Mark attendance using facial recognition
  - Period-wise attendance tracking
  - View attendance statistics
  - Track present and absent students

- **Student Information**
  - List all students in a class
  - Fetch detailed student information
  - View attendance history

## Technical Implementation

### Face Recognition Pipeline
1. **Face Detection**: Uses OpenCV's Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) for initial face detection
2. **Face Encoding**: Utilizes the `face_recognition` library to generate 128-dimensional face encodings
3. **Face Comparison**: Compares face encodings using Euclidean distance with a similarity threshold of 0.5
4. **Attendance Marking**: Automatically marks attendance for recognized students

### Database Structure
The system uses SQLite with the following tables:
- `classes`: Stores class information (id, class_name)
- `students`: Stores student details and face encodings (id, class_id, name, age, email, image, face_encoding)
- `attendance`: Records attendance data (id, student_id, class_id, date, period, status)

## Prerequisites

- Python 3.8 or higher
- Webcam (for attendance marking)
- Internet connection (for first-time setup)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Prince9193/face-recognition-attendance.git
cd smart-attendance
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run student_registration.py
```

2. Access the application:
   - Open your web browser
   - Navigate to `http://localhost:8501`

3. Initial Setup:
   - Create a class
   - Register students with their photos
   - Start marking attendance

## Features in Detail

### Class Management
- Create new classes
- Select class for operations
- Manage multiple classes

### Student Registration
- Enter student details (name, age, email)
- Upload student photo
- Automatic face encoding

### Attendance Marking
- Select class and period
- Upload group photo
- Automatic student recognition
- View present/absent students
- Period-wise attendance tracking

### Attendance Viewing
- View daily attendance
- See attendance statistics
- Track student attendance history

## Troubleshooting

### Face Recognition Issues
- Ensure good lighting conditions
- Use clear, front-facing photos
- Avoid extreme angles
- Make sure faces are clearly visible in group photos

### Database Issues
- If you encounter database errors, use the "Delete Database" option to reset
- The system will automatically recreate the database structure

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [face_recognition](https://github.com/ageitgey/face_recognition) for facial recognition
- [OpenCV](https://opencv.org/) for image processing

## Contact

Prince Singh Rathore

Project Link: https://github.com/Prince9193/face-recognition-attendance.git