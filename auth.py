import streamlit as st

# ✅ Email whitelist (add the allowed users here)
ALLOWED_EMAILS = {
    "youremail@example.com",
    "team@example.com",
    "project.manager@example.com"
}

def login():
    """Displays login UI and validates user email."""
    st.title("🔐 Project Access Login")

    email = st.text_input("Enter your email to access the app:")
    login_button = st.button("Login")

    if login_button:
        if email in ALLOWED_EMAILS:
            st.session_state["authenticated"] = True
            st.session_state["user_email"] = email
            st.success(f"Welcome {email}! Access granted.")
            st.rerun()
        else:
            st.error("🚫 Unauthorized email. Please contact the admin for access.")

def logout():
    """Logs the user out by clearing session data."""
    if "authenticated" in st.session_state:
        del st.session_state["authenticated"]
        del st.session_state["user_email"]
    st.rerun()

def check_auth():
    """Check if user is authenticated."""
    return st.session_state.get("authenticated", False)