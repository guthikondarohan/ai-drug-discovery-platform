# Script to add 5 new views to app_premium.py
import re

# Read the original file
with open('app_premium.py', 'r', encoding='utf-8') as f:
    content = f.read()

# First, update the navigation radio buttons
old_nav = '''    view = st.radio(
        "Navigation",
        ["🧪 Predict", "📊 Analytics", "⭐ Favorites", "📜 History", "🔀 Data Fusion"],
        label_visibility="collapsed"
    )'''

new_nav = '''    view = st.radio(
        "Navigation",
        ["🧪 Predict", "📊 Analytics", "⭐ Favorites", "📜 History", "🔀 Data Fusion",
         "📚 Literature", "⚖️ Compare", "🔍 Explain", "🔎 Search", "🧬 Generate"],
        label_visibility="collapsed"
    )'''

content = content.replace(old_nav, new_nav)

# Add the elif blocks for session state
old_elif = '''    elif view == "🔀 Data Fusion":
        st.session_state.current_view = 'fusion'
    
    st.markdown("---")'''

new_elif = '''    elif view == "🔀 Data Fusion":
        st.session_state.current_view = 'fusion'
    elif view == "📚 Literature":
        st.session_state.current_view = 'literature'
    elif view == "⚖️ Compare":
        st.session_state.current_view = 'compare'
    elif view == "🔍 Explain":
        st.session_state.current_view = 'explain'
    elif view == "🔎 Search":
        st.session_state.current_view = 'search'
    elif view == "🧬 Generate":
        st.session_state.current_view = 'generate'
    
    st.markdown("---")'''

content = content.replace(old_elif, new_elif)

# Now append the 5 new views before the footer
footer_start = '\n# Footer\nst.markdown("---")'

# Read the new views from the demo app
with open('app_features_demo.py', 'r', encoding='utf-8') as f:
    demo_content = f.read()

# Extract just the 5 new view implementations
import_start = demo_content.find('elif feature == "📚 Literature Search":')
import_end = demo_content.find('\n# Footer')
new_views_section = demo_content[import_start:import_end]

# Replace 'feature ==' with 'st.session_state.current_view =='
new_views_section = new_views_section.replace('elif feature ==', 'elif st.session_state.current_view ==')
new_views_section = new_views_section.replace('"📚 Literature Search"', "'literature'")
new_views_section = new_views_section.replace('"⚖️ Molecule Comparison"', "'compare'")
new_views_section = new_views_section.replace('"🔍 Explainable AI"', "'explain'")
new_views_section = new_views_section.replace('"🔎 Similarity Search"', "'search'")
new_views_section = new_views_section.replace('"🧬 Molecule Generation"', "'generate'")

# Insert before footer
content = content.replace(footer_start, '\n' + new_views_section + '\n' + footer_start)

# Write back
with open('app_premium.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Successfully added 5 new features to app_premium.py!")
