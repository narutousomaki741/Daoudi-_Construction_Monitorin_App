# auth.py (PROFESSIONALLY ENHANCED)
import streamlit as st
import logging
from typing import Optional, Dict, Tuple
import hashlib
import time
import jwt
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ProfessionalAuthManager:
    """Professional authentication manager with enhanced security"""
    
    def __init__(self, secret_key: str = "your-secret-key-change-in-production"):
        self.secret_key = secret_key
        self.users = self._load_users()
        self.session_timeout = timedelta(hours=8)  # 8-hour session
    
    def _load_users(self) -> Dict:
        """Load users with role-based access (in production, use database)"""
        return {
            "admin": {
                "password": self._hash_password("admin123"),
                "role": "admin",
                "permissions": ["read", "write", "delete", "admin"]
            },
            "project_manager": {
                "password": self._hash_password("pm123"),
                "role": "project_manager", 
                "permissions": ["read", "write"]
            },
            "viewer": {
                "password": self._hash_password("viewer123"),
                "role": "viewer",
                "permissions": ["read"]
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Secure password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, username: str, role: str) -> str:
        """Generate JWT token for session management"""
        payload = {
            'username': username,
            'role': role,
            'exp': datetime.utcnow() + self.session_timeout,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            st.error("🔐 Session expired. Please log in again.")
            return None
        except jwt.InvalidTokenError:
            st.error("🔐 Invalid session. Please log in again.")
            return None
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """Enhanced authentication with role management"""
        try:
            if username in self.users:
                user_data = self.users[username]
                if user_data["password"] == self._hash_password(password):
                    # Generate token and store session
                    token = self._generate_token(username, user_data["role"])
                    
                    st.session_state.update({
                        'authenticated': True,
                        'username': username,
                        'user_role': user_data["role"],
                        'permissions': user_data["permissions"],
                        'login_time': datetime.now(),
                        'auth_token': token,
                        'session_expiry': datetime.now() + self.session_timeout
                    })
                    
                    logger.info(f"User {username} ({user_data['role']}) authenticated successfully")
                    return True, f"Welcome {username} ({user_data['role']})!"
            
            logger.warning(f"Failed authentication attempt for user: {username}")
            return False, "Invalid username or password"
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, f"Authentication error: {str(e)}"
    
    def require_auth(self, required_permission: str = "read"):
        """Enhanced authentication requirement with permission checking"""
        # Check if session is still valid
        if st.session_state.get('session_expiry'):
            if datetime.now() > st.session_state['session_expiry']:
                st.session_state.clear()
                st.error("🔐 Your session has expired. Please log in again.")
                self.show_login_form()
                st.stop()
        
        if not st.session_state.get('authenticated', False):
            st.warning("🔒 Authentication Required")
            self.show_login_form()
            st.stop()
        
        # Check permissions
        user_permissions = st.session_state.get('permissions', [])
        if required_permission not in user_permissions:
            st.error(f"🚫 Access Denied: You need '{required_permission}' permission to access this page.")
            st.stop()
    
    def show_login_form(self):
        """Professional login form with enhanced UI"""
        st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            st.markdown("### 🔐 Construction Pro Login")
            st.markdown("Please authenticate to access the project management system.")
            
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(
                    "👤 Username",
                    placeholder="Enter your username",
                    help="Contact administrator if you need an account"
                )
                
                password = st.text_input(
                    "🔒 Password", 
                    type="password",
                    placeholder="Enter your password",
                    help="Passwords are case-sensitive"
                )
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    login_button = st.form_submit_button(
                        "🚀 Login", 
                        use_container_width=True,
                        type="primary"
                    )
                with col2:
                    if st.form_submit_button("🔄 Clear", use_container_width=True):
                        st.rerun()
                
                if login_button:
                    if not username or not password:
                        st.error("❌ Please enter both username and password")
                    else:
                        success, message = self.authenticate(username, password)
                        if success:
                            st.success(f"✅ {message}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            # Demo credentials
            with st.expander("🧪 Demo Credentials"):
                st.markdown("""
                **For testing purposes:**
                - **Admin**: `admin` / `admin123`
                - **Project Manager**: `project_manager` / `pm123`  
                - **Viewer**: `viewer` / `viewer123`
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def show_user_profile(self):
        """Show user profile and session information"""
        if st.session_state.get('authenticated'):
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 👤 User Profile")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("🟢")
                with col2:
                    st.markdown(f"**{st.session_state.username}**")
                    st.caption(f"Role: {st.session_state.user_role}")
                
                # Session info
                login_time = st.session_state.get('login_time', datetime.now())
                session_duration = datetime.now() - login_time
                st.caption(f"Session: {str(session_duration).split('.')[0]}")
                
                # Logout button
                if st.button("🚪 Logout", use_container_width=True):
                    self.logout()
    
    def logout(self):
        """Professional logout handling"""
        username = st.session_state.get('username', 'Unknown')
        st.session_state.clear()
        st.success(f"👋 Goodbye {username}! You have been logged out successfully.")
        time.sleep(2)
        st.rerun()
    
    def has_permission(self, permission: str) -> bool:
        """Check if current user has specific permission"""
        return permission in st.session_state.get('permissions', [])

# Global auth instance
auth_manager = ProfessionalAuthManager()

# Convenience functions for easy integration
def require_login(permission: str = "read"):
    """Decorator-style authentication requirement"""
    auth_manager.require_auth(permission)

def show_auth_ui():
    """Show authentication UI components"""
    auth_manager.show_user_profile()
