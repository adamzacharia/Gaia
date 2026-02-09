"""
GaiaChat Streamlit Application

A modern chat interface for exploring Gaia DR3 data using natural language.
Inspired by modern AI chat interfaces with welcome sections and feature cards.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import GaiaChatAgent
from core.config import config
from visualization.plots import (
    create_hr_diagram,
    create_sky_map,
    create_velocity_plot,
    create_toomre_diagram,
    create_proper_motion_plot,
    create_interactive_hr_diagram,
    create_interactive_velocity_plot
)


# Page configuration
st.set_page_config(
    page_title="GaiaChat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with glassmorphism / liquid glass aesthetic
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container - dark gradient background */
    .main {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Sidebar styling - glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Welcome section */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .welcome-icon {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 50%;
        margin: 0 auto 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.4);
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Feature cards - glassmorphism, clickable */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Style buttons inside feature cards to look like the card itself */
    .card-button > button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100%;
        text-align: left;
        color: inherit !important;
    /* Style buttons in example query section as cards */
    /* Hide select buttons under query cards */
    /* robustly target the element container wrapping the query card */
    div[data-testid="element-container"]:has(.query-card) {
        margin-bottom: 0.5rem !important;
    }
    
    /* Target the button container that follows the query card container */
    div[data-testid="element-container"]:has(.query-card) + div[data-testid="element-container"] .stButton {
        margin-top: 0;
        height: auto;
        opacity: 1; /* Make visible */
    }
    
    div[data-testid="element-container"]:has(.query-card) + div[data-testid="element-container"] .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="element-container"]:has(.query-card) + div[data-testid="element-container"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Query card hover effects */
    .query-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.2);
    }
    
    [data-testid="stHorizontalBlock"] .stButton > button strong {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 0.5rem !important;
        display: block;
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-icon.blue { background: rgba(59, 130, 246, 0.2); }
    .feature-icon.purple { background: rgba(139, 92, 246, 0.2); }
    .feature-icon.green { background: rgba(16, 185, 129, 0.2); }
    .feature-icon.orange { background: rgba(245, 158, 11, 0.2); }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.875rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    
    /* Chat messages - glassmorphism */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stChatMessage p {
        color: #e2e8f0 !important;
    }
    
    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.2) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar buttons - glassmorphism */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.2s ease;
        text-align: left;
        padding: 0.75rem 1rem;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateX(4px);
    }
    /* Force equal height columns */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    /* Main area buttons - styled as glassmorphism cards */
    .stButton {
        flex: 1;
        display: flex;
    }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        font-weight: 400;
        padding: 1.25rem 1rem !important;
        transition: all 0.3s ease;
        height: 160px !important;
        min-height: 160px !important;
        max-height: 160px !important;
        text-align: left;
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 0.9rem;
        flex: 1;
        overflow: hidden;
    }
    
    /* Colorful gradient headings in buttons */
    .stButton > button strong {
        display: block;
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.2);
    }
    
    /* Chat input - floating glassmorphism */
    /* Chat input - floating glassmorphism */
    .stChatInput {
        position: fixed;
        bottom: 1rem;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100% - 24rem - 4rem);
        max-width: 800px;
        z-index: 1000;
        background: transparent !important;
    }
    
    /* Input container styling - make it look like a single pill */
    .stChatInput > div {
        background: rgba(30, 41, 59, 0.95) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 9999px !important;
        padding: 0.3rem 0.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide the inner input decoration/borders */
    .stChatInput textarea, .stChatInput input {
        background: transparent !important;
        color: #e2e8f0 !important;
        border: none !important;
        box-shadow: none !important;
        caret-color: #60a5fa;
    }
    
    .stChatInput textarea:focus, .stChatInput input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Send button styling */
    .stChatInput button {
        background: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
    }
    
    .stChatInput button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }
    
    /* Remove any default streamlit input borders */
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
    }
    
    /* Add padding at bottom for floating input */
    .main .block-container {
        padding-bottom: 6rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Data tables - glassmorphism */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Code blocks */
    code {
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
        border-radius: 6px;
        padding: 0.2rem 0.4rem;
    }
    
    /* Expander - glassmorphism */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        font-weight: 500;
        color: #e2e8f0 !important;
    }
    
    details {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
    
    /* Info/Warning boxes - glassmorphism */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.05);
    }
    
    /* Logo text in sidebar */
    .sidebar-logo {
        font-size: 1.75rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 0.25rem;
    }
    
    .sidebar-tagline {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = GaiaChatAgent()
    
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
    
    if "show_data" not in st.session_state:
        st.session_state.show_data = False
    
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True


def render_sidebar():
    """Render the sidebar with navigation and options."""
    with st.sidebar:
        # Logo and branding
        st.markdown('<div class="sidebar-logo">✦ GaiaChat</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Explore Gaia DR3 Data</div>', unsafe_allow_html=True)
        st.divider()
        
        
        
        # Visualization options
        st.markdown("##### Visualizations")
        
        if st.session_state.current_data is not None and len(st.session_state.current_data) > 0:
            plot_type = st.selectbox(
                "Plot Type",
                ["HR Diagram", "Sky Map", "Velocity Plot", "Toomre Diagram", "Proper Motion"],
                key="plot_select",
                label_visibility="collapsed"
            )
            
            if st.button("Generate Plot", use_container_width=True):
                return f"show_plot_{plot_type.lower().replace(' ', '_')}"
            
            st.divider()
            
            # Data view toggle
            st.session_state.show_data = st.checkbox(
                "Show Data Table",
                value=st.session_state.show_data
            )
            
            # Export option
            if st.button("Export CSV", use_container_width=True):
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button(
                    "Download",
                    csv,
                    "gaia_results.csv",
                    "text/csv",
                    use_container_width=True
                )
        else:
            st.info("Run a query to see visualizations")
        
        # Spacer
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        
        # Bottom section
        st.divider()
        
        # New Chat button
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.clear_history()
            st.session_state.current_data = None
            st.session_state.show_welcome = True
            st.rerun()
        
        # About section
        with st.expander("About"):
            st.markdown("""
            **GaiaChat** translates natural language into ADQL queries
            for the Gaia DR3 stellar catalog.
            
            Made by **Adam Zacharia Anil**  
            adamanil@mit.edu
            """)
    
    return None


def render_welcome():
    """Render the welcome screen with feature cards."""
    # Welcome header
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">✦</div>
        <h1 class="welcome-title">Welcome to GaiaChat</h1>
        <p class="welcome-subtitle">
            Explore the Gaia DR3 stellar catalog using natural language. 
            Ask questions about stars, their positions, velocities, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards with exact query prompts - using HTML for fixed height
    st.markdown("### Example queries to get started")
    
    # Define queries with their headings - all with same display height
    cards = [
        ("Solar Neighborhood", "Show me the nearest 100 stars"),
        ("High Velocity Stars", "Find stars within 3 kpc with total space velocities exceeding 350 km/s"),
        ("Cone Search", "Search for stars around RA=180, Dec=45 with a 2 degree radius"),
        ("Bright Stars", "Find bright stars with parallax greater than 10 mas"),
    ]
    
    # Create 4 equal-width columns
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    for i, (title, query) in enumerate(cards):
        with cols[i]:
            # Fixed height HTML card - Visual layer with blue Run button
            st.markdown(f'''
            <div style="
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 1.25rem;
                height: 180px;
                display: flex;
                flex-direction: column;
                transition: all 0.3s ease;
                position: relative;
            " class="query-card">
                <div style="
                    font-size: 1.1rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 0.5rem;
                ">{title}</div>
                <div style="
                    color: #94a3b8;
                    font-size: 0.85rem;
                    line-height: 1.5;
                    flex: 1;
                    overflow: hidden;
                    margin-bottom: 0.5rem;
                ">"{query}"</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Button below the card
            if st.button("Run", key=f"card_{i}", use_container_width=True):
                st.session_state.show_welcome = False
                return query
    
    return None


def render_plot(plot_type: str):
    """Render a plot based on the plot type."""
    if st.session_state.current_data is None:
        st.warning("No data available. Run a query first.")
        return
    
    df = st.session_state.current_data
    
    try:
        if plot_type == "hr_diagram":
            fig = create_hr_diagram(df)
            st.pyplot(fig)
        elif plot_type == "sky_map":
            fig = create_sky_map(df)
            st.pyplot(fig)
        elif plot_type == "velocity_plot":
            if 'V_R' in df.columns and 'V_phi' in df.columns:
                fig = create_velocity_plot(df)
                st.pyplot(fig)
            else:
                st.warning("Velocity data not available. Run a kinematic query first.")
        elif plot_type == "toomre_diagram":
            if all(col in df.columns for col in ['V_R', 'V_phi', 'V_z']):
                fig = create_toomre_diagram(df)
                st.pyplot(fig)
            else:
                st.warning("Full velocity data not available.")
        elif plot_type == "proper_motion":
            fig = create_proper_motion_plot(df)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")


def process_query(prompt: str):
    """Process a user query and display the response."""
    # Hide welcome screen
    st.session_state.show_welcome = False
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Querying Gaia archive..."):
            try:
                response = st.session_state.agent.chat(prompt)
                
                # Display response
                st.markdown(response.message)
                
                # Store data if available
                if response.data is not None and len(response.data) > 0:
                    st.session_state.current_data = response.data
                    
                    # Show data preview
                    with st.expander("View Data Sample"):
                        st.dataframe(response.data.head(10))
                    
                    # Show ADQL query if available
                    if response.query_used:
                        with st.expander("ADQL Query"):
                            st.code(response.query_used, language="sql")
                
                # Auto-generate relevant plot if suggested
                if response.plot_type:
                    st.divider()
                    st.markdown("### Visualization")
                    render_plot(response.plot_type)
                
                # Store message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.message,
                    "data": response.data,
                    "plot_type": response.plot_type,
                    "query_used": response.query_used
                })
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Render sidebar and get any selected suggestion
    sidebar_action = render_sidebar()
    
    # Handle sidebar actions
    if sidebar_action:
        if sidebar_action.startswith("show_plot_"):
            plot_type = sidebar_action.replace("show_plot_", "")
            # Add plot as a new assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Here is the {plot_type.replace('_', ' ')} based on current data.",
                "plot_type": plot_type
            })
            st.rerun()
        else:
            # It's a query
            process_query(sidebar_action)
            st.rerun()
    
    # Show welcome screen or chat
    if st.session_state.show_welcome and len(st.session_state.messages) == 0:
        welcome_action = render_welcome()
        if welcome_action:
            process_query(welcome_action)
            st.rerun()
    else:
        # Header for chat mode
        st.markdown("## GaiaChat")
        st.markdown("*Ask me anything about the Gaia stellar catalog*")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show data preview if available
                if message["role"] == "assistant" and "data" in message and message["data"] is not None:
                    with st.expander("View Data Sample"):
                        st.dataframe(message["data"].head(10))
                
                # Show ADQL Query if available
                if message.get("query_used"):
                    with st.expander("ADQL Query"):
                        st.code(message["query_used"], language="sql")
                
                # Show plot if requested
                if message["role"] == "assistant" and "plot_type" in message and message["plot_type"]:
                    render_plot(message["plot_type"])
        
        # Data table display
        if st.session_state.show_data and st.session_state.current_data is not None:
            st.divider()
            st.markdown("### Current Data")
            st.dataframe(
                st.session_state.current_data,
                use_container_width=True,
                height=300
            )
    
    # Check for pending query
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        process_query(query)
    
    # Chat input (always visible)
    if prompt := st.chat_input("Ask about Gaia data..."):
        process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
