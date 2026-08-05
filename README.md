# 🚦 Smart Traffic Violation Detection System

An AI-powered application that automatically detects traffic violations using computer vision and OCR technology. Built specifically for Indian traffic scenarios. Includes both a **modern web dashboard** (`index.html`) and a **Streamlit backend app** (`app.py`).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)

## 🎯 Core Features

### 🔍 Violation Detection
- **Helmetless Riding** 🪖❌ - Detects riders without helmets
- **Triple Riding** 🏍️ - Identifies more than 2 persons on two-wheelers
- **Signal Jumping** 🚦❌ - Detects red signal violations
- **Over-speeding** 🏎️💨 - Identifies speeding vehicles

### 📸 Advanced Capabilities
- **Number Plate OCR** - Extracts Indian vehicle registration numbers
- **Real-time Analysis** - Process images instantly
- **High Accuracy** - ~70-85% detection confidence
- **Database Storage** - SQLite/CSV storage for violation records
- **Web Dashboard** - Interactive frontend with live map, charts, and notifications
- **Professional UI** - Blue-themed modern interface

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Core Language | Python 3.8+ |
| Computer Vision | OpenCV |
| OCR Engine | Tesseract (pytesseract) |
| Web Framework | Streamlit |
| Web Dashboard | HTML + Vanilla JS + CSS |
| Database | SQLite + CSV |
| Data Processing | Pandas, NumPy |
| Image Processing | Pillow |

## 📁 Project Structure

```
Smart-traffic-Management/
│
├── index.html                 # Web dashboard (frontend entry point)
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration & constants
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── core/                      # Core modules
│   ├── __init__.py
│   ├── detector.py            # Violation detection logic
│   ├── ocr.py                 # Number plate OCR
│   ├── utils.py               # Helper functions
│   └── database.py            # Database manager
│
├── css/                       # Stylesheet assets
│   └── style.css              # Custom styles
│
├── js/                        # JavaScript modules
│   ├── app.js                 # Main application logic
│   ├── charts.js              # Charts & analytics
│   ├── data.js                # Data management
│   ├── map.js                 # Live map integration
│   └── notifications.js       # Notification system
│
└── data/                      # Data storage
    └── samples/               # Sample test images
```

## 📦 Installation

### Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Tesseract OCR**
   
   **Windows:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to PATH
   - Or download installer: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.0.20221214.exe
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/invo-coder19/Smart-traffic-Management.git
   cd Smart-traffic-Management
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **Windows:**
   ```bash
   .\venv\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Option A: Web Dashboard (Frontend)

Simply open `index.html` in your browser — no server required:
```bash
start index.html       # Windows
open index.html        # macOS
xdg-open index.html    # Linux
```

The dashboard provides:
- Live traffic map with incident markers
- Real-time charts and analytics
- Notification panel for active violations

### Option B: Streamlit App (Backend)

1. **Activate virtual environment** (if not already activated)
   ```bash
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. **Run Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Access the application**
   - Open browser at: `http://localhost:8501`

### Using the Streamlit Application

#### 1. Detection Page
- Upload a traffic image (JPG, JPEG, PNG)
- Click "Detect Violations"
- View detection results and confidence scores
- Check extracted number plate (if visible)
- Save record to database

#### 2. Database Page
- View all violation records
- Filter by violation type or number plate
- Export data to CSV

#### 3. Statistics Page
- View total violations and fines
- Analyze violations by type and severity
- Track detection confidence metrics

#### 4. About Page
- Learn about the system
- View violation types and fines
- Check system requirements

### Violation Display
- Color-coded severity levels (High, Medium, None)
- Confidence scores with visual indicators
- Annotated images with bounding boxes
- Detailed violation information

## 📊 Detection Accuracy

| Violation Type | Accuracy | Method |
|---------------|----------|---------| 
| Helmetless | ~70-80% | Color-based head detection |
| Triple Riding | ~65-75% | Contour analysis & person counting |
| Signal Jump | ~60-70% | Red signal & motion detection |
| Over-speeding | ~50-60% | Motion blur analysis (simulated) |
| Number Plate OCR | ~75-85% | Tesseract with pattern matching |

*Note: Accuracy depends on image quality, lighting, and angle*

## 🔧 Configuration

### Adjusting Detection Thresholds

Edit `config.py`:

```python
# Detection Thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence
HELMET_COLOR_THRESHOLD = 0.3          # Helmet detection sensitivity
PERSON_COUNT_THRESHOLD = 2            # Max persons on two-wheeler
SPEED_LIMIT_KMH = 40                  # Speed limit
```

### Database Selection

In `app.py`, change database type:

```python
# Use SQLite
st.session_state.database = ViolationDatabase(use_sqlite=True)

# Or use CSV
st.session_state.database = ViolationDatabase(use_sqlite=False)
```

## 🐛 Troubleshooting

### Tesseract not found
**Error:** `pytesseract.pytesseract.TesseractNotFoundError`

**Solution:** 
1. Install Tesseract OCR
2. Add to PATH or set path in `core/ocr.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### OpenCV import error
**Error:** `ImportError: No module named cv2`

**Solution:**
```bash
pip install opencv-python
```

### Low detection accuracy
**Solutions:**
- Use higher quality images
- Ensure good lighting
- Avoid extreme angles
- Use images with clear subjects
- Adjust thresholds in `config.py`

### Database locked error
**Solution:**
- Close other instances of the app
- Delete `data/violations.db` and restart

## 🚀 Future Enhancements

### Planned Features
- [ ] Deep learning models (YOLO, SSD)
- [ ] Real-time video processing
- [ ] Multiple camera support
- [ ] Cloud deployment
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

### ML Model Integration
Currently using rule-based CV. To integrate ML:
1. Train custom models on labeled traffic datasets
2. Use pre-trained models (YOLO, Faster R-CNN)
3. Replace detection methods in `core/detector.py`

## 📄 License

This project is open source and available for educational and research purposes.

## 👥 Use Cases

- **Traffic Police**: Automate violation detection
- **Smart Cities**: Integrate with CCTV systems
- **Highways**: Monitor high-speed corridors
- **Research**: Study traffic patterns
- **Safety Campaigns**: Raise awareness

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- Tesseract OCR for text recognition
- Streamlit for rapid web app development
- Indian traffic authorities for inspiration


---

<div align="center">

🚦 Stay Safe | Follow Rules | Save Lives 🚦

</div>
