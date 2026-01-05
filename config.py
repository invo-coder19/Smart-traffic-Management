"""
Configuration file for Smart Traffic Violation Detection System
Contains UI colors, violation types, and system settings
"""

# UI Color Scheme - Blue Theme
PRIMARY_BLUE = "#1E3A8A"      # Dark Blue
ACCENT_BLUE = "#3B82F6"       # Medium Blue
LIGHT_BLUE = "#60A5FA"        # Light Blue
SKY_BLUE = "#BAE6FD"          # Sky Blue
SUCCESS_GREEN = "#10B981"
WARNING_YELLOW = "#F59E0B"
DANGER_RED = "#EF4444"
NEUTRAL_GRAY = "#6B7280"

# Violation Types
VIOLATION_TYPES = {
    "HELMETLESS": {
        "name": "Helmetless Riding 🪖❌",
        "description": "Rider not wearing helmet",
        "severity": "HIGH",
        "fine": 1000
    },
    "TRIPLE_RIDING": {
        "name": "Triple Riding 🏍️",
        "description": "More than 2 persons on a two-wheeler",
        "severity": "MEDIUM",
        "fine": 1000
    },
    "SIGNAL_JUMP": {
        "name": "Signal Jumping 🚦❌",
        "description": "Crossing red signal",
        "severity": "HIGH",
        "fine": 1000
    },
    "OVER_SPEEDING": {
        "name": "Over-speeding 🏎️💨",
        "description": "Exceeding speed limit",
        "severity": "HIGH",
        "fine": 2000
    },
    "NO_VIOLATION": {
        "name": "No Violation ✅",
        "description": "Traffic rules followed",
        "severity": "NONE",
        "fine": 0
    }
}

# Detection Thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for detection
HELMET_COLOR_THRESHOLD = 0.3         # Threshold for helmet detection
PERSON_COUNT_THRESHOLD = 2            # Max persons allowed on two-wheeler
SPEED_LIMIT_KMH = 40                  # Default speed limit in km/h

# OCR Settings
OCR_LANGUAGE = "eng"
INDIAN_PLATE_PATTERN = r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}"

# Database Settings
DB_NAME = "data/violations.db"
CSV_NAME = "data/violations.csv"

# Image Settings
MAX_IMAGE_SIZE = (1280, 720)  # Max size for processing
UPLOAD_FORMATS = ["jpg", "jpeg", "png"]

# App Settings
APP_TITLE = "🚦 Smart Traffic Violation Detection System"
APP_ICON = "🚦"
SIDEBAR_STATE = "expanded"
