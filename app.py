import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from google import genai
from google.genai import types
from rhythm_mapping import classify_rhythm_state # Import the classification logic

# --- 1. LLM Setup ---
def load_gemini_key():
    """Tries to load the API key from st.secrets, raising a clear error if missing."""
    try:
        # Standard Streamlit location (preferred)
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
        
        raise Exception("GEMINI_API_KEY not found. Please ensure 'secrets.toml' is in the '.streamlit' folder or set as an environment variable.")

try:
    GEMINI_API_KEY = load_gemini_key()
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error loading API Key: {e}")
    st.stop()
# --- End LLM Setup ---


# --- 2. Define the Rhythm States and Theme (for display purposes) ---
RHYTHM_STATES = {
    "Flow State": {"productivity": "High", "screen_time": "Low", "sleep": "High", "color": "#00A39C", "icon": "⚡"},
    "Digital Drift State": {"productivity": "Medium", "screen_time": "High", "sleep": "Medium", "color": "#8058a5", "icon": "🌀"},
    "Balanced State": {"productivity": "Medium", "screen_time": "Medium", "sleep": "Medium", "color": "#306DCC", "icon": "⚖️"},
    "Burnout State": {"productivity": "Low", "screen_time": "High", "sleep": "Low", "color": "#C44E52", "icon": "🔥"},
    "Fatigue State": {"productivity": "Medium", "screen_time": "Medium", "sleep": "Low", "color": "#FF8C00", "icon": "😴"},
    "Unknown State": {"productivity": "N/A", "screen_time": "N/A", "sleep": "N/A", "color": "gray", "icon": "❓"},
}

# --- 3. Gemini Prompt Generation ---
def generate_llm_prompt(state_name, user_data):
    """Creates a detailed prompt for the Gemini LLM."""
    
    state_details = RHYTHM_STATES.get(state_name, RHYTHM_STATES["Unknown State"])
    
    ml_insights_context = (
        "Context from Data Analysis: Screen time is the primary disruptor to productivity, "
        "and prioritizing rest is the most effective way to sustain flow. "
        "The intervention must be non-digital and human-centered."
    )
    
    prompt = f"""
    You are 'Rhythm AI', an empathetic, human-centered productivity coach. Your role is to translate data patterns into personalized, non-technical advice for a user.

    **User's Current State (Clustering Result):** {state_name}
    **State Characteristics:** {state_details['productivity']} Productivity, {state_details['screen_time']} Screen Time, {state_details['sleep']} Sleep.
    **User's Today's Goal:** {user_data['goal']}
    **User's Metrics:** {user_data['screen_time']} hours screen time, {user_data['sleep_hours']} hours sleep, {user_data['productivity_score']}% productivity.

    {ml_insights_context}

    **Your Task:**
    1.  **Insight:** Write a short, empathetic paragraph (max 3 sentences) acknowledging their current state and explaining what it means for their goal in simple, human language. Use the cluster label (e.g., 'Flow State') once.
    2.  **Microbreak:** Generate one personalized, non-digital microbreak (2-5 minutes) specifically tailored to address the core need of their state, focusing on reflection, breathing, or planning.
    3.  **Format:** Output the response in JSON format with two keys: 'insight' and 'microbreak'.
    """
    return prompt

# --- 4. Helper: Render the Input UI section (Matching Figma) ---
def render_input_ui():
    
    # Ensure session state variables for inputs exist
    if "sleep_input" not in st.session_state:
        st.session_state["sleep_input"] = 7.0
    if "screen_time_input" not in st.session_state:
        st.session_state["screen_time_input"] = 8.0
    if "productivity_slider" not in st.session_state:
        st.session_state["productivity_slider"] = 60
    if "goal_text" not in st.session_state:
        st.session_state["goal_text"] = "Finish presentation deck"
    
    # --- 2-Column Layout for Metrics ---
    col1, col2 = st.columns(2)

    # Metric 1: Sleep
    with col1:
        st.markdown('<div class="metric-label">Sleep Hours</div>', unsafe_allow_html=True)
        st.session_state["sleep_input"] = st.number_input(
            "Sleep_Hidden_Label_1", 
            min_value=0.0, max_value=24.0, value=st.session_state["sleep_input"], 
            step=0.25, key="sleep_key", format="%.2f",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Metric 2: Screen Time
    with col2:
        st.markdown('<div class="metric-label">Screen Time</div>', unsafe_allow_html=True)
        st.session_state["screen_time_input"] = st.number_input(
            "Screen_Time_Hidden_Label_2", 
            min_value=0.0, max_value=24.0, value=st.session_state["screen_time_input"], 
            step=0.25, key="screen_time_key", format="%.2f",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Productivity Score Slider (Full Width below the row) ---
    st.markdown('<div class="productivity-header">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">How productive do you feel now?</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state["productivity_slider"] = st.slider(
        "Productivity_Hidden_Label_3", 
        0, 100, st.session_state["productivity_slider"], 5, key="productivity_key",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Goal Card (Text Area) ---
    st.markdown('<div style="font-weight:700; color:#262626; margin-bottom:12px;">Today\'s Key Goal / Focus</div>', unsafe_allow_html=True)
    
    st.session_state["goal_text"] = st.text_area(
        "Goal_Hidden_Label_4", 
        value=st.session_state["goal_text"], 
        key="goal_key", 
        placeholder="What would you like to accomplish today?", 
        height=140,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True) # close goal-card
    
    # --- CTA Button ---
    st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
    analyze_clicked = st.button("analyze rhythm", key="analyze_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    return {
        "sleep_hours": st.session_state["sleep_input"],
        "screen_time": st.session_state["screen_time_input"],
        "productivity_score": st.session_state["productivity_slider"],
        "goal": st.session_state["goal_text"],
        "analyze_clicked": analyze_clicked
    }


# --- 5. Main Application Flow ---

st.set_page_config(page_title="Rhythm Dashboard", layout="centered")

# --- Load Figtree font & core CSS (Figma styling) ---
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
    :root{
        --bg: #faf9f7;
        --card-bg: #ffffff;
        --muted: #7a726c;
        --accent: #262626;
        --soft-shadow: 0 6px 14px rgba(34,34,34,0.06);
        --card-radius: 16px;
        --container-width: 900px;
    }
    html, body, .stApp {
        background: var(--bg);
        font-family: 'Figtree', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .main .block-container {
        max-width: var(--container-width) !important;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 2rem; 
    }
    .hero {
        text-align: center;
        margin-bottom: 36px;
    }
    .hero h1 {
        font-size: 72px;
        margin: 0;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -1px;
    }
    .hero p {
        margin-top: 8px;
        color: #4f4a48;
        font-size: 40px;
        font-weight: 600;
    }

    .metric-card {
        background: var(--card-bg);
        border-radius: var(--card-radius);
        border: #e8e6e1;
        box-shadow: var(--soft-shadow);
        padding: 20px;
        min-height: 140px; 
    }
    .metric-label { 
        font-weight: 600; 
        margin-bottom: 12px; 
    }
    .productivity-card {
        background: var(--card-bg);
        border-radius: var(--card-radius);
        border: #e8e6e1;
        box-shadow: var(--soft-shadow);
        padding: 20px;
        margin-top: 20px;
    }
    .productivity-header {
        display: flex;
        justify-content: space-between;     
        align-items: center;
        margin-bottom: 12px;
    }
    .goal-card {
        background: var(--card-bg);
        border-radius: var(--card-radius);
        border: #e8e6e1;
        box-shadow: var(--soft-shadow);
        padding: 20px;
        margin-top: 20px;
    }
    .cta-wrap {
        display: flex;
        justify-content: center;
        margin-top: 24px;
        margin-bottom: 40px;
    }
    .cards-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
    }
    .result-card {
        background: var(--card-bg);
        border-radius: var(--card-radius);
        border: #e8e6e1;
        box-shadow: var(--soft-shadow);
        padding: 20px;
        display: flex;
        align-items: flex-start;
    }    
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Centered Title and Subtitle (Hero) ---
st.markdown(
    """
    <div class="hero">
        <h1>rhythm</h1>
        <p>a mindful productivity dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# --- End Centered Title ---


# Initialize analysis state
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'input_expanded' not in st.session_state:
    st.session_state.input_expanded = True 

# Use a main container for the inputs so we can selectively hide/show it
if st.session_state.analyze_clicked and not st.session_state.input_expanded:
    # If analysis has run AND inputs are collapsed, create an expander for the inputs
    with st.expander("Show/Modify Inputs"):
        inputs_container = st.container()
        with inputs_container:
            inputs = render_input_ui()
            # If the analyze button is clicked from inside the expander, re-run analysis
            if inputs["analyze_clicked"]:
                st.session_state.user_inputs = inputs
                st.session_state.analyze_clicked = True
                st.session_state.input_expanded = False
                st.rerun() 
            # Button to un-collapse the main input view
            if st.button("Un-Collapse Inputs to Main View", key="uncollapse_btn"):
                st.session_state.input_expanded = True
                st.session_state.analyze_clicked = False
                st.rerun()
                
    # Use the stored inputs for analysis
    inputs = st.session_state.get("user_inputs", {})
    
else:
    # Input section is fully visible
    inputs_container = st.container()
    with inputs_container:
        inputs = render_input_ui()
        
        # If Analyze is clicked, store inputs and change state
        if inputs["analyze_clicked"]:
            st.session_state.user_inputs = inputs
            st.session_state.analyze_clicked = True
            st.session_state.input_expanded = False 
            st.rerun() 

# --- Analysis & Results Logic (Only runs if a valid analysis has been triggered) ---

if st.session_state.analyze_clicked and inputs:
    
    sleep_hours = inputs['sleep_hours']
    screen_time = inputs['screen_time']
    productivity_score = inputs['productivity_score']
    
    # Step 1: Classify User State using ML Logic
    state_name = classify_rhythm_state(sleep_hours, screen_time, productivity_score)
    
    if "Error" in state_name or state_name == "Unknown State":
        st.error(f"ML Classification Error: State could not be determined. Result: {state_name}")
        # Offer button to show inputs again
        if st.button("Modify Inputs", key="error_modify_btn"):
            st.session_state.input_expanded = True
            st.session_state.analyze_clicked = False
            st.rerun()
        st.stop()
    
    state_info = RHYTHM_STATES[state_name]
    state_color = state_info['color']
    
    # Step 2: Generate LLM Prompt
    llm_prompt = generate_llm_prompt(state_name, inputs)

    # Step 3: Call Gemini API
    st.markdown(
        f"""
        <div class="results-hero">
            <h2>Your Rhythm Analysis</h2>
            <p>Based on your inputs, here's how to optimize your day</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner(f"Analyzing your {state_name} rhythm and generating microbreak..."):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=llm_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "insight": {"type": "string"},
                            "microbreak": {"type": "string"},
                        },
                        "required": ["insight", "microbreak"]
                    }
                )
            )
            
            llm_results = json.loads(response.text)
            
            # --- START RESULTS LAYOUT (Matching Figma Mockup) ---
            
            # 1. Status Bar (Full Width Card - Top)
            st.markdown(f'''
                <div class="productivity-card" style="margin-top:8px; border-left: 4px solid {state_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-weight:800; font-size:18px;">CURRENT RHYTHM:</div>
                        <div style="font-weight:800; font-size:24px; color:{state_color};">{state_info['icon']} {state_name}</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            # 2. ROW 1: Insight and Microbreak (2-column grid)
            st.markdown('<div class="cards-grid">', unsafe_allow_html=True)

            # Card A: LLM Insight
            st.markdown(f'''
                <div class="result-card">
                    <div class="result-icon" style="background:{state_color}1A; color:{state_color};"></div>
                    <div class="result-content">
                        <h3>LLM Insight</h3>
                        <p>{llm_results.get('insight','')}</p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Card B: Microbreak
            st.markdown(f'''
                <div class="result-card" style="margin-top:20px;">
                    <div class="result-icon" style="background:{state_color}1A; color:{state_color};"></div>
                    <div class="result-content">
                        <h3>Focus Action</h3>
                        <p>{llm_results.get('microbreak','')}</p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True) # close cards-grid

            # 3. Inputs for transparency (Full width card - Bottom)
            st.markdown(f'''
                <div class="productivity-card" style="margin-top:20px; padding:20px;">
                    <div style="font-weight:700; color:var(--accent); margin-bottom:8px;">Inputs for Transparency</div>
                    <div style="font-size:20px; color:var(--muted);">
                        Sleep: {sleep_hours} hrs | 
                        Screen Time: {screen_time} hrs | 
                        Productivity: {productivity_score}% | 
                        Goal: {inputs['goal']}
                    </div>
                    <div style="margin-top:15px;">
                        <p style="font-size:14px; color:var(--muted); margin:0;">
                            *The classification is based on historical patterns. Review the data or modify inputs to see changes.
                        </p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Explicit button to return to modify inputs on the main page
            st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
            if st.button("Start New Analysis", key="new_analysis_btn"):
                st.session_state.input_expanded = True
                st.session_state.analyze_clicked = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while calling the Gemini API. Error: {e}")
            if st.button("Modify Inputs (API Error)", key="api_error_modify_btn"):
                st.session_state.input_expanded = True
                st.session_state.analyze_clicked = False
                st.rerun()


st.markdown("""
<style>

:root {
    --bg: #f8f6f2;
    --card-bg: #ffffff;
    --accent: #8a7d68;
    --text: #2b2b2b;
    --card-radius: 16px;
    --soft-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

/* overall page */
body {
    background: var(--bg);
}

/* ---------- CARD LOOK FOR STREAMLIT WIDGETS ---------- */
.stNumberInput, .stSlider, .stTextArea, .stTextInput {
    background: var(--card-bg) !important;
    border-radius: var(--card-radius) !important;
    padding: 16px !important;
    box-shadow: var(--soft-shadow) !important;
    border: 1px solid #e8e6e1 !important;
    margin-bottom: 8px !important;
}

/* widget label styling */
.stNumberInput > label,
.stSlider > label,
.stTextArea > label,
.stTextInput > label {
    margin-bottom: 6px !important;
    font-weight: 600 !important;
    color: var(--accent) !important;
}

/* slider spacing fix */
.stSlider > div[role="slider"] {
    margin-top: 8px !important;
}

/* center-align button */
.stButton > button {
    border-radius: 12px !important;
    padding: 8px 24px !important;
    font-weight: 700 !important;
    background: var(--accent) !important;
    color: white !important;
}

/* header */
.header-title {
    font-size: 32px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 4px;
    color: var(--text);
}

.header-sub {
    font-size: 16px;
    color: var(--accent);
    margin-bottom: 22px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# ✨ INPUT UI COMPONENT (fixed)
# -----------------------------
def render_input_ui():
    st.session_state.setdefault("sleep", 7.0)
    st.session_state.setdefault("screen", 8.0)
    st.session_state.setdefault("prod", 60)
    st.session_state.setdefault("goal", "Finish presentation deck")

    # layout: sleep + screen in one row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='metric-label'>Sleep Hours</div>", unsafe_allow_html=True)
        st.session_state.sleep = st.number_input(
            "Sleep",
            min_value=0.0, max_value=24.0,
            step=0.25,
            value=st.session_state.sleep,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<div class='metric-label'>Screen Time</div>", unsafe_allow_html=True)
        st.session_state.screen = st.number_input(
            "ScreenTime",
            min_value=0.0, max_value=24.0,
            step=0.25,
            value=st.session_state.screen,
            label_visibility="collapsed"
        )

    st.markdown("<div class='metric-label' style='margin-top:16px;'>How productive do you feel right now?</div>", unsafe_allow_html=True)
    st.session_state.prod = st.slider(
        "ProdSlider",
        0, 100,
        value=st.session_state.prod,
        step=5,
        label_visibility="collapsed"
    )

    return {
        "sleep": st.session_state.sleep,
        "screen_time": st.session_state.screen,
        "productivity": st.session_state.prod,
        "goal": st.session_state.goal,
        "clicked": analyze_clicked
    }