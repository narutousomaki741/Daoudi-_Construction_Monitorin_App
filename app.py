import streamlit as st
import logging
import yaml
import os
from datetime import datetime
import time

# Try to load config, use defaults if not available
try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Default config if file doesn't exist
    CONFIG = {
        'app': {'name': 'Construction Project Manager', 'environment': 'development'},
        'ui': {'theme': 'light', 'default_zones': 2}
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

def main():
    """Main application with authentication"""
    try:
        # Professional setup
        setup_page_config()
        
        # Initialize auth manager with config
        from auth import auth_manager, show_auth_ui
        auth_manager.secret_key = CONFIG['auth']['secret_key']
        auth_manager.session_timeout = timedelta(hours=CONFIG['auth']['session_timeout_hours'])
        
        # Check authentication for all pages
        auth_manager.require_auth("read")
        
        # Inject professional CSS
        from ui_pages import inject_professional_css
        inject_professional_css()
        
        # Render professional sidebar with user profile
        from ui_pages import render_enhanced_sidebar
        mission = render_enhanced_sidebar()
        
        # Show user profile in sidebar
        show_auth_ui()
        
        # Main content area
        st.markdown(f'<div class="main-header">🏗️ {CONFIG["app"]["name"]}</div>', 
                   unsafe_allow_html=True)
        
        # Route to appropriate page with permission checks
        if "🚀 Generate Schedule" in mission:
            if auth_manager.has_permission("write"):
                from ui_pages import generate_schedule_ui
                generate_schedule_ui()
            else:
                st.error("🚫 You need 'write' permission to generate schedules")
                
        elif "📊 Monitor Project" in mission:
            from ui_pages import monitor_project_ui
            monitor_project_ui()
            
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Professional error handling
        st.error(f"""
        🚨 **System Error Occurred**
        
        We encountered an unexpected error. Please:
        
        1. Refresh the page and try again
        2. Check your input data for formatting issues  
        3. Contact support if the problem persists
        
        **Error Details:** `{str(e)}`
        """)
        
        # Show technical details in expander
        with st.expander("🔧 Technical Details (For Support)"):
            st.exception(e)

if __name__ == "__main__":
    main()
