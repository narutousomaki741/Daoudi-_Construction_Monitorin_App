import streamlit as st
import logging
import yaml
import os
from datetime import datetime, timedelta
import time

# Try to load config, use defaults if not available
try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Default config if file doesn't exist
    CONFIG = {
        'app': {
            'name': 'Construction Project Manager Pro', 
            'version': '2.0.0',
            'environment': 'development',
            'description': 'Professional construction scheduling and monitoring tool'
        },
        'ui': {
            'theme': 'light', 
            'default_zones': 2,
            'page_layout': 'wide'
        },
        'auth': {
            'secret_key': 'construction-pro-default-secret-key',
            'session_timeout_hours': 8
        }
    }

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, CONFIG.get('logging', {}).get('level', 'INFO')),
    format=CONFIG.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

def setup_page_config():
    """Professional page configuration with theme support"""
    st.set_page_config(
        page_title=CONFIG['app']['name'],
        layout=CONFIG['ui'].get('page_layout', 'wide'),
        page_icon="🏗️",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/docs',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': f"### {CONFIG['app']['name']} v{CONFIG['app']['version']}\n{CONFIG['app']['description']}"
        }
    )

def inject_app_css():
    """Inject CSS that complements ui_pages.py styles"""
    st.markdown("""
    <style>
    /* Main header styling that matches ui_pages.py theme */
    .app-main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 1rem;
    }
    
    /* App description */
    .app-description {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Sidebar enhancements */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Global button enhancements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def render_app_sidebar():
    """Simple sidebar that works with your ui_pages.py help section"""
    with st.sidebar:
        # Header with logo
        st.markdown("""
        <div class="sidebar-header">
            <h1 style="color: white; margin-bottom: 0.5rem;">🏗️</h1>
            <h3 style="color: white; margin: 0;">Construction Pro</h3>
            <p style="color: #ccc; margin: 0; font-size: 0.9rem;">v{CONFIG['app']['version']}</p>
        </div>
        """.format(CONFIG=CONFIG), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Mission selection - SIMPLE VERSION (matches your ui_pages.py)
        mission = st.radio(
            "**🎯 Select Mission**",
            ["Generate Schedule", "Monitor Project"],
            index=0,
            help="Choose between creating a new schedule or monitoring existing project progress"
        )
        
        st.markdown("---")
        
        # Environment info
        environment = CONFIG['app'].get('environment', 'development')
        env_color = "🟢" if environment == "production" else "🟡"
        
        with st.expander("🌐 System Info", expanded=False):
            st.write(f"**Environment:** {env_color} {environment}")
            st.write(f"**Version:** {CONFIG['app'].get('version', '2.0.0')}")
            st.write(f"**Last refresh:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, help="Refresh the application"):
                st.rerun()
        with col2:
            if st.button("🧹 Clear", use_container_width=True, help="Clear temporary data"):
                keys_to_keep = ['session_start', 'zones_floors', 'start_date']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                st.success("Cache cleared!")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            <p>Professional Edition</p>
        </div>
        """, unsafe_allow_html=True)
    
    return mission

def main():
    """Main application with proper error handling"""
    try:
        # Professional setup
        setup_page_config()
        inject_app_css()
        
        # Initialize session start time
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        
        # Try to initialize auth manager (optional)
        auth_available = False
        try:
            from auth import auth_manager, show_auth_ui
            # Configure auth manager if config available
            if 'auth' in CONFIG:
                auth_manager.secret_key = CONFIG['auth'].get('secret_key', 'default-secret-key')
                auth_manager.session_timeout = timedelta(
                    hours=CONFIG['auth'].get('session_timeout_hours', 8)
                )
            auth_available = True
        except ImportError:
            st.info("🔓 **Note:** Authentication module not available - running in public mode")
        
        # Apply authentication if available
        if auth_available:
            try:
                auth_manager.require_auth("read")
                show_auth_ui()
            except Exception as auth_error:
                st.warning(f"⚠️ Authentication issue: {auth_error}")
                # Continue without auth
                auth_available = False
        
        # Render sidebar
        mission = render_app_sidebar()
        
        # Main content area
        app_name = CONFIG['app']['name']
        st.markdown(f'<div class="app-main-header">🏗️ {app_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="app-description">{CONFIG["app"]["description"]}</div>', unsafe_allow_html=True)
        
        # Add environment indicator
        environment = CONFIG['app'].get('environment', 'development')
        if environment != 'production':
            env_color = "orange" if environment == "staging" else "blue"
            st.markdown(f"""
            <div style="background-color: {env_color}; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; margin-bottom: 1rem;">
                <strong>{environment.upper()} ENVIRONMENT</strong> - For testing purposes only
            </div>
            """, unsafe_allow_html=True)
        
        # Inject UI styles from ui_pages.py (this function EXISTS in your ui_pages.py)
        from ui_pages import inject_ui_styles
        inject_ui_styles()
        
        # Route to appropriate page with permission checks
        if mission == "Generate Schedule":
            # Check permissions if auth is available
            if auth_available and not auth_manager.has_permission("write"):
                st.error("""
                🚫 **Access Denied**
                
                You need **write permission** to generate schedules.
                
                **Available accounts with write access:**
                - **Project Manager:** `project_manager` / `pm123`
                - **Admin:** `admin` / `admin123`
                
                Please logout and login with appropriate credentials.
                """)
            else:
                # Import and run the schedule UI (this function EXISTS in your ui_pages.py)
                from ui_pages import generate_schedule_ui
                generate_schedule_ui()
                
        elif mission == "Monitor Project":
            # Import and run the monitoring UI (this function EXISTS in your ui_pages.py)
            from ui_pages import monitor_project_ui
            monitor_project_ui()
        
        # Add global footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
            <p>Built with Streamlit • {app_name} v{version}</p>
        </div>
        """.format(
            app_name=CONFIG['app']['name'],
            version=CONFIG['app'].get('version', '2.0.0')
        ), unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Professional error handling
        st.error("""
        🚨 **Application Error**
        
        We encountered an unexpected error. This is usually temporary.
        
        **Please try:**
        1. Refreshing the page
        2. Checking your internet connection  
        3. Trying again in a few moments
        
        If the problem persists, please contact support.
        """)
        
        # Technical details in expander
        with st.expander("🔧 Technical Details for Support"):
            st.exception(e)
            
            # System information
            st.markdown("**System Information:**")
            st.code(f"""
            Timestamp: {datetime.now()}
            Environment: {CONFIG['app'].get('environment', 'unknown')}
            Session Keys: {list(st.session_state.keys())}
            """)

if __name__ == "__main__":
    main()
