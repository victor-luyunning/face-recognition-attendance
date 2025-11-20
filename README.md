# Photo-based Attendance Marking System

An intelligent attendance management system based on classroom group photos, using advanced facial recognition technology to automate the attendance process. Built with Streamlit and Python, this system provides a zero-hardware-cost solution for educational institutions, achieving imperceptible attendance marking in real classroom scenarios.

> 🎓 **Academic Project**: SEU Pattern Classification Course Project  
> 👥 **Authors**: Duan Jiawen (71123329), Yang Xirui (71123331)  
> 🏫 **Institution**: Southeast University  
> 📅 **Date**: November 2025

## ✨ Key Features

### 🎯 Core Innovations
- **Multi-Photo Attendance**: Upload multiple classroom group photos for comprehensive coverage
- **Robust Face Detection**: Triple detection mechanism (HOG → CNN → Enhanced) with 96.7% detection rate
- **Automatic Deduplication**: Maximum similarity algorithm prevents duplicate marking across photos
- **Optimized Threshold**: Custom threshold (0.42) balancing 87.8% precision and 90% recall
- **Zero Hardware Cost**: Pure software solution, CPU-only, no GPU required

### 📋 Management Features
- **Class Management**: Create and manage multiple classes
- **Student Registration**: Register students with frontal photos and automatic face encoding
- **Batch Attendance**: Upload 5-10 classroom photos, auto-recognize all students in 8.7 seconds
- **Period Tracking**: Support 9 periods per day with automatic absence marking
- **Visual Reports**: Real-time attendance statistics and exportable CSV reports
- **CRUD Operations**: Complete student information management (Create, Read, Update, Delete)

## 🔧 Technical Implementation

### Advanced Face Recognition Pipeline
1. **Image Preprocessing**: Auto-scaling to 1600px max dimension for performance optimization
2. **Triple Face Detection**:
   - **Layer 1**: HOG model (fast, ~68% baseline)
   - **Layer 2**: CNN model fallback (if HOG fails)
   - **Layer 3**: Contrast enhancement + 2x upsampling (final rescue)
3. **Face Encoding**: 128-dimensional feature vectors using dlib's ResNet-based model
4. **Similarity Matching**: Euclidean distance with optimized threshold of 0.42 for classroom scenarios
5. **Deduplication**: Maximum similarity selection across multiple photos per student

### Database Schema (SQLite)
```sql
-- classes: Class information
(id INTEGER PRIMARY KEY, class_name TEXT UNIQUE)

-- students: Student details with face encodings
(id, class_id, name, age, email UNIQUE, image BLOB, face_encoding BLOB)

-- attendance: Attendance records with duplicate prevention
(id, student_id, class_id, date, period, status)
-- UNIQUE constraint on (student_id, date, period)
```

### System Architecture
- **Frontend**: Streamlit (responsive web interface)
- **Backend**: Python 3.10
- **Database**: SQLite (embedded, zero-config)
- **CV Libraries**: OpenCV 4.8 + face_recognition 1.3.0 (dlib 19.24)
- **Deployment**: Tencent Cloud (4-core, 4GB RAM, Ubuntu 22.04, port 8501)

## 📋 Prerequisites

- **Python**: 3.10 or higher (3.7+ compatible)
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS
- **RAM**: Minimum 4GB (8GB+ recommended for large classes)
- **Storage**: 500MB for dependencies + database
- **Camera**: Optional (for live photo capture, can use uploaded photos)
- **Internet**: Required for initial package installation

## 🚀 Installation

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/victor-luyunning/face-recognition-attendance.git
cd face-recognition-attendance
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

> ⚠️ **Note**: Installing `dlib` and `face_recognition` may take 5-10 minutes on first install.  
> If you encounter CMake errors on Windows, install Visual Studio Build Tools first.

### Alternative: Using Conda (Recommended)
```bash
conda create -n attendance python=3.10
conda activate attendance
pip install -r requirements.txt
```

## 💻 Usage

### 1. Start the Application
```bash
streamlit run student_registration.py
```

### 2. Access the Web Interface
- **Local**: Open browser and navigate to `http://localhost:8501`
- **Remote**: Access via `http://<your-server-ip>:8501`

### 3. Workflow

#### Initial Setup (One-time)
1. **Create Class**: Navigate to sidebar → "Create Class" → Enter class name
2. **Register Students**: 
   - Select class
   - Go to "Register Student" tab
   - Upload clear frontal photo + enter details (name, age, email)
   - Repeat for all students (supports batch registration)

#### Daily Attendance
1. **Select Class & Period**: Choose from sidebar
2. **Upload Photos**: Go to "Mark Attendance" tab
   - Upload 5-10 classroom group photos (different angles recommended)
   - Support JPG/PNG, max 10MB per photo
3. **Auto Recognition**: System processes all photos in ~8 seconds
4. **Review Results**: 
   - View recognized students with similarity scores
   - Check absent student list
   - Export to CSV if needed

#### View Reports
- **View Attendance**: Check period-wise or daily statistics
- **Export Data**: Download attendance records as CSV
- **Fetch Student Details**: Query individual attendance history

## 📊 Performance Metrics

Tested on real university classroom (40-55 students, 9 class sessions):

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 87.8% | Correctly recognized / Total recognized |
| **Recall** | 90.0% | Correctly recognized / Actually present |
| **F1 Score** | 88.9% | Harmonic mean of precision and recall |
| **Detection Rate** | 96.7% | Face detection success rate |
| **Processing Time** | 8.7s | Average time per class session (5-10 photos) |
| **False Positive** | <1.8% | Incorrect recognition rate |

### Threshold Selection Rationale
| Threshold | Precision | Recall | F1 | Notes |
|-----------|-----------|--------|-----|-------|
| 0.60 (default) | 95.2% | 72.5% | 82.4% | High miss rate |
| 0.50 | 92.3% | 85.0% | 88.5% | Balanced |
| **0.42 (ours)** | **87.8%** | **90.0%** | **88.9%** | **Optimal for classroom** |
| 0.38 | 82.1% | 92.5% | 87.0% | Too many false positives |

## 🛠️ Troubleshooting

### Face Detection Failures
- **Symptom**: "No faces detected in photo"
- **Solutions**:
  - Ensure adequate lighting (avoid strong backlighting)
  - Upload photos from multiple angles (front, side, back rows)
  - Check image resolution (min 800×600, max 6000×4000)
  - Avoid group photos with <50px face size

### Low Recognition Rate
- **Symptom**: Many students marked absent but actually present
- **Solutions**:
  - Re-register students with higher quality frontal photos
  - Upload more group photos (5-10 from different angles)
  - Check lighting consistency between registration and attendance photos
  - Verify students aren't wearing masks or hats

### Duplicate Recognition
- **Symptom**: Same student marked multiple times
- **Status**: ✅ Automatically handled by deduplication algorithm
- The system keeps only the highest similarity match

### Database Issues
- **Reset Database**: Use "Delete Database" in sidebar (⚠️ irreversible)
- **Backup**: Database stored in `students.db`, can be manually copied

### Performance Issues
- **Slow Processing**: Reduce photo resolution or number of photos
- **Memory Error**: Close other applications, use smaller batch size

### Installation Errors
```bash
# dlib installation error on Windows
pip install cmake
pip install dlib

# face_recognition installation error
pip install --no-cache-dir face_recognition
```

## 📁 Project Structure

```
face-recognition-attendance/
│
├── student_registration.py            # Main Streamlit application
├── evaluate_accuracy.py               # Evaluation tool for testing
├── requirements.txt                   # Python dependencies
├── students.db                        # SQLite database (auto-created)
│
├── test_data/                         # Test photos for evaluation
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
│
├── ground_truth.json                  # Ground truth labels for evaluation
├── evaluation_report_*.json           # Evaluation results
├── *_export.csv                       # Exported attendance reports
│
├── Task2_P02_Project_Report_zh_CN.pdf # Project report PDF (Chinese)
├── Task2_P02_Project_Report.pdf       # Project report PDF 
└── README.md                          # This file
```

## 🧪 Evaluation Tool

The project includes an automated evaluation script:

```bash
python evaluate_accuracy.py
```

This will:
1. Process all photos in `test_data/` directory
2. Compare results with `ground_truth.json`
3. Calculate precision, recall, F1, and accuracy
4. Generate detailed JSON report

## 📚 Documentation

- **Academic Report**: See `Task2_P02_Project_Report.pdf` or `Task2_P02_Project_Report_zh_CN.pdf` for detailed system design and evaluation
- **API Reference**: All functions documented in `student_registration.py`
- **Database Schema**: See Section 4.3 in report

## 🤝 Contributing

This is an academic project. For suggestions or improvements:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Original Project**: [Prince9193/face-recognition-attendance](https://github.com/Prince9193/face-recognition-attendance)
- **Libraries**:
  - [Streamlit](https://streamlit.io/) - Web interface framework
  - [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library (dlib wrapper)
  - [OpenCV](https://opencv.org/) - Computer vision and image processing
  - [dlib](http://dlib.net/) - Machine learning toolkit
- **Course**: SEU Pattern Classification (Fall 2025)
- **Institution**: Southeast University

## 📧 Contact

- **Jiawen Duan**: 1146102617@qq.com
- **Xirui Yang**: 3245105379@qq.com
- **Project Link**: https://github.com/victor-luyunning/face-recognition-attendance

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐!
