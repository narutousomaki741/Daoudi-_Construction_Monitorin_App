import streamlit as st
import logging
from datetime import datetime
import time

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_page_config():
    """Professional page configuration"""
    st.set_page_config(
        page_title="🏗️ Construction Project Manager Pro",
        layout="wide",
        page_icon="🏗️",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/docs',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': "### Construction Project Management Tool v2.0\nProfessional scheduling and monitoring solution."
        }
    )

def inject_professional_css():
    """Inject professional CSS styling"""
    st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        padding: 1rem;
    }
    
    /* Professional cards */
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .professional-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    /* Enhanced metric cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    /* Button enhancements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50, #3498db);
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Professional sidebar with session management"""
    with st.sidebar:
        # Header with logo
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: white; margin-bottom: 0.5rem;">🏗️</h1>
            <h3 style="color: white; margin: 0;">Construction Pro</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        # Mission selection
        st.markdown("### 🎯 Mission Control")
        mission = st.radio(
            "Select Mission",
            ["🚀 Generate Schedule", "📊 Monitor Project"],
            index=0,
            help="Choose between creating a new schedule or monitoring existing project progress"
        )
        st.markdown("---")
        # Session information
        st.markdown("### 📈 Session Info")
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        session_duration = datetime.now() - st.session_state.session_start
        st.info(f"**Session active:** {str(session_duration).split('.')[0]}")
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.clear()
                st.success("Cache cleared!")
                time.sleep(1)
                st.rerun()
        st.markdown("---")
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            <p>Construction Manager Pro v2.0</p>
            <p>© 2024 Your Company</p>
        </div>
        """, unsafe_allow_html=True)
    return mission

def main():
    """Main application entry point"""
    try:
        # Professional setup
        setup_page_config()
        inject_professional_css()
        # Render professional sidebar
        mission = render_sidebar() 
        # Main content area
        st.markdown('<div class="main-header">🏗️ Construction Project Manager Pro</div>', 
                   unsafe_allow_html=True)
        # Route to appropriate page
        if "🚀 Generate Schedule" in mission:
            from ui_pages import generate_schedule_ui
            generate_schedule_ui()
        else:
            from ui_pages import monitor_project_ui
            monitor_project_ui()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Professional error handling
        st.error("""
        🚨 **System Error Occurred**
        We encountered an unexpected error. Please:
        
        1. Refresh the page and try again
        2. Check your input data for formatting issues  
        3. Contact support if the problem persists
        
        **Error Details:** `{}`
        """.format(str(e)))
        
        # Show technical details in expander
        with st.expander("🔧 Technical Details (For Support)"):
            st.exception(e)

if __name__ == "__main__":
    main()
