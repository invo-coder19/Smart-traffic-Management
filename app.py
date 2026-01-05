"""
Smart Traffic Violation Detection System - Streamlit App
Modern UI with fixed layout to prevent screen shifting
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime

# Import core modules
import config
from core.detector import ViolationDetector
from core.ocr import NumberPlateOCR
from core.utils import ImageProcessor
from core.database import ViolationDatabase


# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state=config.SIDEBAR_STATE
)

# Modern CSS with animations and fixed layout
def load_custom_css():
    """Load modern blue-themed CSS with animations"""
    css = f"""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main theme colors */
    :root {{
        --primary-blue: {config.PRIMARY_BLUE};
        --accent-blue: {config.ACCENT_BLUE};
        --light-blue: {config.LIGHT_BLUE};
        --sky-blue: {config.SKY_BLUE};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Smooth transitions */
    .stApp {{
        transition: all 0.3s ease;
    }}
    
    /* Modern Header with animation */
    .main-header {{
        background: linear-gradient(135deg, {config.PRIMARY_BLUE} 0%, {config.ACCENT_BLUE} 50%, {config.LIGHT_BLUE} 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(30, 58, 138, 0.3);
        animation: slideDown 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(1.1); opacity: 0.8; }}
    }}
    
    @keyframes slideDown {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .main-header p {{
        margin: 0.8rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }}
    
    /* Modern Card Design */
    .modern-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }}
    
    .modern-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Violation Card with gradient border */
    .violation-card {{
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, {config.ACCENT_BLUE}, {config.LIGHT_BLUE}) border-box;
        border: 3px solid transparent;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
    }}
    
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    .violation-card h3 {{
        color: {config.PRIMARY_BLUE};
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 700;
    }}
    
    .violation-card p {{
        margin: 0.5rem 0;
        line-height: 1.6;
        color: #374151;
    }}
    
    /* Modern Stat Boxes with glassmorphism */
    .stat-box {{
        background: linear-gradient(135deg, {config.LIGHT_BLUE} 0%, {config.SKY_BLUE} 100%);
        backdrop-filter: blur(10px);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: {config.PRIMARY_BLUE};
        box-shadow: 0 8px 20px rgba(96, 165, 250, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: scaleIn 0.5s ease-out;
    }}
    
    @keyframes scaleIn {{
        from {{ opacity: 0; transform: scale(0.9); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    .stat-box:hover {{
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 12px 30px rgba(96, 165, 250, 0.4);
    }}
    
    .stat-box h2 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {config.PRIMARY_BLUE}, {config.ACCENT_BLUE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .stat-box p {{
        margin: 0.8rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.85;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Modern Button Styling */
    .stButton>button {{
        background: linear-gradient(135deg, {config.ACCENT_BLUE} 0%, {config.PRIMARY_BLUE} 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {config.PRIMARY_BLUE} 0%, #1e40af 100%);
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }}
    
    .stButton>button:active {{
        transform: translateY(0);
    }}
    
    /* Severity Badges with animation */
    .severity-high {{
        background: linear-gradient(135deg, {config.DANGER_RED}, #dc2626);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
        animation: pulse-badge 2s ease-in-out infinite;
    }}
    
    .severity-medium {{
        background: linear-gradient(135deg, {config.WARNING_YELLOW}, #f59e0b);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }}
    
    .severity-none {{
        background: linear-gradient(135deg, {config.SUCCESS_GREEN}, #059669);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }}
    
    @keyframes pulse-badge {{
        0%, 100% {{ box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3); }}
        50% {{ box-shadow: 0 4px 15px rgba(239, 68, 68, 0.5); }}
    }}
    
    /* Upload section styling */
    .uploadedFile {{
        border: 2px dashed {config.ACCENT_BLUE};
        border-radius: 10px;
        padding: 1rem;
    }}
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning {{
        border-radius: 10px;
        animation: slideIn 0.3s ease-out;
    }}
    
    /* Fixed container to prevent shifting */
    .fixed-container {{
        min-height: 600px;
        transition: all 0.3s ease;
    }}
    
    /* Image container with fixed aspect ratio */
    .image-container {{
        position: relative;
        width: 100%;
        min-height: 400px;
        border-radius: 15px;
        overflow: hidden;
        background: #f3f4f6;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {config.PRIMARY_BLUE} 0%, {config.ACCENT_BLUE} 100%);
        padding: 2rem 1rem;
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    /* Modern section headers */
    .section-header {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {config.PRIMARY_BLUE};
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {config.LIGHT_BLUE};
        display: inline-block;
    }}
    
    /* Loading spinner custom color */
    .stSpinner > div {{
        border-top-color: {config.ACCENT_BLUE} !important;
    }}
    
    /* Dataframe styling */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
    }}
    
    /* Chart containers */
    .stPlotlyChart {{
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = ViolationDetector()
    
    if 'ocr' not in st.session_state:
        st.session_state.ocr = NumberPlateOCR()
    
    if 'database' not in st.session_state:
        st.session_state.database = ViolationDatabase(use_sqlite=True)
    
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None


# Main application
def main():
    """Main application function"""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize
    init_session_state()
    
    # Modern Header with animation
    st.markdown(f"""
    <div class="main-header">
        <h1>{config.APP_ICON} Smart Traffic Violation Detection</h1>
        <p>🤖 AI-Powered Traffic Monitoring • 🎯 Real-time Detection • 🔒 Secure Database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with gradient background
    with st.sidebar:
        st.markdown("# 🎯 Navigation")
        page = st.radio(
            "",
            ["🔍 Detection", "💾 Database", "📊 Statistics", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Min Confidence",
            0.0, 1.0, config.DETECTION_CONFIDENCE_THRESHOLD,
            help="Minimum confidence for violation detection"
        )
        
        config.DETECTION_CONFIDENCE_THRESHOLD = confidence_threshold
    
    # Route to pages
    if page == "🔍 Detection":
        detection_page()
    elif page == "💾 Database":
        database_page()
    elif page == "📊 Statistics":
        statistics_page()
    else:
        about_page()


def detection_page():
    """Main detection page with fixed layout"""
    
    # Create fixed containers to prevent shifting
    upload_container = st.container()
    results_container = st.container()
    
    with upload_container:
        st.markdown('<p class="section-header">📷 Upload Traffic Image</p>', unsafe_allow_html=True)
        
        # Use columns with fixed layout
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image (JPG, JPEG, PNG)",
                type=config.UPLOAD_FORMATS,
                help="Upload a traffic image to detect violations",
                key="file_uploader"
            )
            
            # Fixed image container
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="📸 Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detection button
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    detect_btn = st.button("🔍 Detect Violations", use_container_width=True, type="primary")
                with col_btn2:
                    if st.button("🗑️ Clear", use_container_width=True):
                        st.session_state.detection_results = None
                        st.session_state.uploaded_image = None
                        st.rerun()
                
                if detect_btn:
                    with st.spinner("🔄 Analyzing image..."):
                        # Run detection
                        detection_results = st.session_state.detector.detect_violations(image)
                        
                        # Run OCR
                        ocr_results = st.session_state.ocr.extract_plate(image)
                        
                        # Store results
                        st.session_state.detection_results = {
                            'detection': detection_results,
                            'ocr': ocr_results,
                            'image': image
                        }
                    
                    st.success("✅ Analysis complete!")
            else:
                # Placeholder when no image
                st.markdown("""
                <div class="image-container">
                    <div style="text-align: center; padding: 2rem;">
                        <p style="font-size: 3rem; margin: 0;">📷</p>
                        <p style="color: #6B7280; margin-top: 1rem;">Upload an image to get started</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Fixed results container - always rendered
            st.markdown('<div class="fixed-container">', unsafe_allow_html=True)
            
            if st.session_state.detection_results:
                results = st.session_state.detection_results
                detection = results['detection']
                ocr = results['ocr']
                
                
                st.markdown("### 🎯 Detection Results")
                
                violation_type = detection['primary_violation']
                violation_info = config.VIOLATION_TYPES[violation_type]
                
                severity_class = f"severity-{violation_info['severity'].lower()}"
                
                st.markdown(f"""
                <div class="violation-card">
                    <h3>{violation_info['name']}</h3>
                    <p><strong>📝 Description:</strong> {violation_info['description']}</p>
                    <p><strong>⚠️ Severity:</strong> <span class="{severity_class}">{violation_info['severity']}</span></p>
                    <p><strong>💰 Fine Amount:</strong> ₹{violation_info['fine']:,}</p>
                    <p><strong>🎯 Confidence:</strong> {detection['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("### 🚗 Number Plate Recognition")
                if ocr['detected']:
                    st.success(f"**Plate Number:** `{ocr['plate_text']}`")
                    st.info(f"**OCR Confidence:** {ocr['confidence']:.1%}")
                else:
                    st.warning("⚠️ No number plate detected")
                
                st.markdown("---")
                
                st.markdown("### 📊 Annotated Result")
                annotated = cv2.cvtColor(detection['annotated_image'], cv2.COLOR_BGR2RGB)
                st.image(annotated, use_container_width=True)
                
                st.markdown("---")
                
                # Action buttons
                col_save, col_new = st.columns([1, 1])
                with col_save:
                    if st.button("💾 Save to Database", use_container_width=True, type="primary"):
                        plate_text = ocr['plate_text'] if ocr['detected'] else 'N/A'
                        
                        st.session_state.database.add_violation(
                            violation_type=violation_type,
                            violation_name=violation_info['name'],
                            number_plate=plate_text,
                            confidence=detection['confidence'],
                            severity=violation_info['severity'],
                            fine=violation_info['fine'],
                            image_path=uploaded_file.name if uploaded_file else 'unknown'
                        )
                        
                        st.success("✅ Record saved to database!")
                        st.balloons()
            else:
                # Placeholder for results
                st.markdown("""
                <div class="modern-card" style="min-height: 400px; display: flex; align-items: center; justify-content: center;">
                    <div style="text-align: center;">
                        <p style="font-size: 3rem; margin: 0;">🎯</p>
                        <p style="color: #6B7280; margin-top: 1rem; font-size: 1.1rem;">Detection results will appear here</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


def database_page():
    """Database viewing page with modern design"""
    st.markdown('<p class="section-header">💾 Violation Database</p>', unsafe_allow_html=True)
    
    # Get all violations
    df = st.session_state.database.get_all_violations(limit=100)
    
    if len(df) > 0:
        # Stats summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <h2>{len(df)}</h2>
                <p>Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_plates = df['number_plate'].nunique()
            st.markdown(f"""
            <div class="stat-box">
                <h2>{unique_plates}</h2>
                <p>Unique Vehicles</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_fines = df['fine'].sum() if 'fine' in df.columns else 0
            st.markdown(f"""
            <div class="stat-box">
                <h2>₹{total_fines:,.0f}</h2>
                <p>Total Fines</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Filter Records")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            violation_filter = st.selectbox(
                "Violation Type",
                ["All"] + list(config.VIOLATION_TYPES.keys())
            )
        
        with col2:
            if 'number_plate' in df.columns:
                plates = df['number_plate'].unique().tolist()
                plate_filter = st.selectbox(
                    "Number Plate",
                    ["All"] + [p for p in plates if p != 'N/A']
                )
        
        with col3:
            if st.button("📥 Export to CSV", use_container_width=True):
                csv_path = f"violations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = st.session_state.database.export_to_csv(csv_path)
                st.success(f"✅ Exported to {filepath}")
        
        
        # Apply filters
        filtered_df = df.copy()
        
        if violation_filter != "All":
            filtered_df = filtered_df[filtered_df['violation_type'] == violation_filter]
        
        if 'plate_filter' in locals() and plate_filter != "All":
            filtered_df = filtered_df[filtered_df['number_plate'] == plate_filter]
        
        st.markdown("---")
        st.markdown(f"### 📋 Records ({len(filtered_df)} entries)")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )
    
    else:
        st.markdown("""
        <div class="modern-card" style="min-height: 300px; display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center;">
                <p style="font-size: 4rem; margin: 0;">📊</p>
                <h3 style="color: #6B7280; margin-top: 1rem;">No Records Yet</h3>
                <p style="color: #9CA3AF;">Upload and detect violations to populate the database</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def statistics_page():
    """Statistics dashboard with modern visualizations"""
    st.markdown('<p class="section-header">📈 Analytics Dashboard</p>', unsafe_allow_html=True)
    
    stats = st.session_state.database.get_statistics()
    
    # Summary stats with modern design
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h2>{stats['total_violations']}</h2>
            <p>Total Violations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h2>₹{stats['total_fines']:,.0f}</h2>
            <p>Total Fines</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h2>{stats['avg_confidence']:.0%}</h2>
            <p>Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <h2>{len(stats['by_type'])}</h2>
            <p>Violation Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts in modern cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Violations by Type")
        if stats['by_type']:
            type_df = pd.DataFrame(list(stats['by_type'].items()), columns=['Type', 'Count'])
            st.bar_chart(type_df.set_index('Type'), height=300)
        else:
            st.info("No data available")
    
    with col2:
        st.markdown("### ⚠️ Violations by Severity")
        if stats['by_severity']:
            severity_df = pd.DataFrame(list(stats['by_severity'].items()), columns=['Severity', 'Count'])
            st.bar_chart(severity_df.set_index('Severity'), height=300)
        else:
            st.info("No data available")


def about_page():
    """About page with modern design"""
    st.markdown('<p class="section-header">ℹ️ System Information</p>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h3>🎯 Core Features</h3>
            <ul style="line-height: 2;">
                <li><strong>Real-time Detection</strong> - Instant analysis</li>
                <li><strong>Multi-Violation</strong> - 4+ violation types</li>
                <li><strong>OCR Integration</strong> - Indian plate recognition</li>
                <li><strong>Database Storage</strong> - SQLite + CSV</li>
                <li><strong>Modern UI</strong> - Responsive design</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card">
            <h3>🛠️ Technology Stack</h3>
            <ul style="line-height: 2;">
                <li><strong>Python 3.8+</strong> - Core language</li>
                <li><strong>OpenCV</strong> - Computer vision</li>
                <li><strong>Tesseract OCR</strong> - Text recognition</li>
                <li><strong>Streamlit</strong> - Web framework</li>
                <li><strong>SQLite</strong> - Database</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Violation Types & Fines")
    
    for key, info in config.VIOLATION_TYPES.items():
        if key != "NO_VIOLATION":
            severity_class = f"severity-{info['severity'].lower()}"
            st.markdown(f"""
            <div style="padding: 1rem; margin: 0.5rem 0; background: #f9fafb; border-radius: 10px; border-left: 4px solid {config.ACCENT_BLUE};">
                <h4 style="margin: 0;">{info['name']}</h4>
                <p style="margin: 0.5rem 0; color: #6B7280;">{info['description']}</p>
                <p style="margin: 0.5rem 0;">
                    <span class="{severity_class}">{info['severity']}</span>
                    <strong style="margin-left: 1rem;">Fine: ₹{info['fine']:,}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="modern-card" style="text-align: center; margin-top: 2rem;">
        <h3>🚦 Making Roads Safer with AI</h3>
        <p>Powered by Computer Vision & Machine Learning</p>
        <p style="color: #6B7280; font-size: 0.9rem; margin-top: 1rem;">
            © 2026 Smart Traffic Violation Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
