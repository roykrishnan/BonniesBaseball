import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import os
import glob

# Page configuration
st.set_page_config(
    page_title="St. Bonaventure Baseball Player Lookup", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: white !important;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: white !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    .player-header {
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid #54342c;
        padding-bottom: 0.5rem;
    }
    .section-header {
        color: #54342c;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid #54342c;
        padding-bottom: 0.5rem;
    }
    .stSelectbox > div > div {
        background-color: #54342c !important;
        border: 2px solid #54342c !important;
    }
    .stSelectbox > div > div > div {
        color: white !important;
    }
    .stSelectbox label {
        color: white !important;
        font-weight: bold !important;
    }
    .stMetric > div {
        background-color: #ffffff;
        border: 1px solid #54342c;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .stMetric .metric-label {
        color: #54342c;
    }
    .stMetric .metric-value {
        color: #54342c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_player_data():
    """Load all player data for lookup functionality"""
    data_dir = "data"
    all_players = {}
    
    if not os.path.exists(data_dir):
        return {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        return {}
    
    # Handedness mapping
    handedness_map = {
        'Andrew Ayers': 'LHP',
        'Tom Wilkie': 'LHP', 
        'Conor Wolf': 'LHP',
        'Jakson Ross': 'LHP',
        'Mark Holm': 'LHP',
        'Nolan Feidt': 'LHP',
        'Ty Corey': 'LHP'
    }
    
    for csv_file in csv_files:
        try:
            # Try different encodings
            encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
            lines = None
            successful_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    successful_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                continue
            
            player_id = None
            player_name = None
            data_start_row = None
            
            # Find player info
            for i, line in enumerate(lines):
                if 'Player ID:' in line:
                    player_id = line.split(',')[1].strip()
                elif 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                elif line.startswith('No,Date'):
                    data_start_row = i
                    break
            
            if data_start_row is not None and player_name and player_id:
                # Read pitch data
                pitch_data = pd.read_csv(csv_file, skiprows=data_start_row, encoding=successful_encoding)
                
                # Filter valid pitches
                pitch_data = pitch_data[pitch_data['Pitch Type'].notna()]
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '-']
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '']
                
                if len(pitch_data) > 0:
                    # Convert numeric columns including Release Side
                    numeric_cols = ['Velocity', 'Total Spin', 'VB (trajectory)', 'HB (trajectory)', 'Release Height', 'Release Side', 'Horizontal Angle']
                    for col in numeric_cols:
                        if col in pitch_data.columns:
                            pitch_data[col] = pd.to_numeric(pitch_data[col], errors='coerce')
                    
                    # Special handling for Gyro Degree column which might have concatenated values
                    if 'Gyro Degree (deg)' in pitch_data.columns:
                        # Handle cases where multiple values might be concatenated
                        def clean_gyro_degree(value):
                            if pd.isna(value) or value == '' or value == '-':
                                return np.nan
                            try:
                                # If it's already a number, return it
                                return float(value)
                            except (ValueError, TypeError):
                                # If it's a string with multiple values, take the first one
                                str_val = str(value)
                                # Split by common separators and take the first numeric value
                                import re
                                numbers = re.findall(r'-?\d+\.?\d*', str_val)
                                if numbers:
                                    return float(numbers[0])
                                return np.nan
                        
                        pitch_data['Gyro Degree (deg)'] = pitch_data['Gyro Degree (deg)'].apply(clean_gyro_degree)
                    
                    # Store player data
                    all_players[player_name] = {
                        'player_id': player_id,
                        'handedness': handedness_map.get(player_name, 'RHP'),
                        'pitch_data': pitch_data,
                        'file_path': csv_file
                    }
        
        except Exception as e:
            continue
    
    return all_players

def calculate_bonnies_stuff_plus_for_pitch_type(df, pitch_type, player_fastball_velocity=None):
    """Calculate Bonnies Stuff+ for a specific pitch type using updated weights"""
    
    def normalize_component(values, higher_is_better=True):
        if len(values) == 0 or values.std() == 0:
            return np.ones(len(values)) * 0.5
        
        # Use z-score normalization (more industry standard)
        z_scores = (values - values.mean()) / values.std()
        
        # Convert to 0-1 scale with sigmoid function
        normalized = 1 / (1 + np.exp(-z_scores))
        
        if not higher_is_better:
            normalized = 1 - normalized
        return normalized
    
    def normalize_deviation_from_mean(values):
        """Reward deviation from mean - both high and low values are good"""
        if len(values) == 0 or values.std() == 0:
            return np.ones(len(values)) * 0.5
        
        # Calculate absolute deviation from mean
        mean_val = values.mean()
        deviations = np.abs(values - mean_val)
        
        # Normalize deviations (higher deviation = better)
        if deviations.std() == 0:
            return np.ones(len(values)) * 0.5
            
        z_scores = (deviations - deviations.mean()) / deviations.std()
        normalized = 1 / (1 + np.exp(-z_scores))
        
        return normalized
    
    # Updated weights - final version
    weights = {
        'velocity': 0.225,         # 22.5%
        'spin_rate': 0.175,        # 17.5%  
        'release_height': 0.125,   # 12.5%
        'distinctive_shape': 0.125, # 12.5%
        'release_side': 0.085,     # 8.5%
        'horizontal_break': 0.10,  # 10%
        'vertical_break': 0.10,    # 10%
        'speed_diff': 0.075,       # 7.5%
        'horizontal_angle': 0.05   # 5%
    }
    
    # Calculate metrics for this pitch type
    velocity = df['Velocity'].mean()
    spin_rate = df['Total Spin'].mean()
    release_height = df['Release Height'].mean() if 'Release Height' in df.columns else 5.5
    release_side = df['Release Side'].mean() if 'Release Side' in df.columns else 0
    horizontal_angle = df['Horizontal Angle'].mean() if 'Horizontal Angle' in df.columns else 0
    h_break = abs(df['HB (trajectory)'].mean()) if 'HB (trajectory)' in df.columns else 0
    v_break = df['VB (trajectory)'].mean() if 'VB (trajectory)' in df.columns else 0
    
    # Calculate speed difference vs player's own fastball
    if pitch_type != 'Fastball' and player_fastball_velocity is not None:
        speed_diff = max(0, player_fastball_velocity - velocity)
    else:
        speed_diff = 0
    
    # For single player analysis, we need to create artificial distributions for normalization
    # Since we only have one data point per metric, we'll use simplified scoring
    
    # Velocity scoring (pitch-type specific)
    if pitch_type == 'Fastball':
        velocity_score = min(1.0, max(0.0, (velocity - 70) / 30))  # 70-100 mph range
    elif pitch_type == 'ChangeUp':
        velocity_score = min(1.0, max(0.0, (velocity - 65) / 25))  # 65-90 mph range
    else:  # Slider, Curveball, etc.
        velocity_score = min(1.0, max(0.0, (velocity - 65) / 30))  # 65-95 mph range
    
    # Spin rate scoring (pitch-type specific)
    if pitch_type == 'Fastball':
        spin_score = min(1.0, max(0.0, (spin_rate - 1800) / 1200))  # 1800-3000 rpm range
    elif pitch_type == 'ChangeUp':
        # For changeups, lower spin can be better, but some spin is needed
        spin_score = max(0.2, min(1.0, 1.0 - (spin_rate - 1000) / 1500))  # Inverted scoring
    else:  # Breaking balls
        spin_score = min(1.0, max(0.0, (spin_rate - 2000) / 1000))  # 2000-3000 rpm range
    
    # Release height deviation (reward uniqueness)
    # Assume team average release height is around 5.5 feet
    team_avg_height = 5.5
    height_deviation = abs(release_height - team_avg_height)
    height_score = min(1.0, height_deviation / 1.0)  # 1 foot deviation gets full points
    
    # Release side deviation (reward uniqueness)
    # Assume team average release side is around 0 feet (center)
    team_avg_side = 0.0
    side_deviation = abs(release_side - team_avg_side)
    side_score = min(1.0, side_deviation / 2.0)  # 2 feet deviation gets full points
    
    # Horizontal angle (lower is better)
    angle_score = max(0.0, 1.0 - (abs(horizontal_angle) / 20))  # 0-20 degrees
    
    # Speed differential (for off-speed pitches)
    speed_diff_score = min(1.0, speed_diff / 15) if pitch_type != 'Fastball' else 0.5  # 15 mph max
    
    # Movement scores
    h_break_score = min(1.0, h_break / 20)  # 20 inches max
    v_break_score = min(1.0, abs(v_break) / 20)  # 20 inches max
    
    # Distinctive shape: reward pitches where |horizontal_break| - |vertical_break| deviates from 0
    shape_differential = abs(h_break - abs(v_break))
    distinctive_shape_score = min(1.0, shape_differential / 15)  # 15 inches max difference
    
    # Calculate weighted composite score
    composite_score = (
        velocity_score * weights['velocity'] +
        spin_score * weights['spin_rate'] +
        height_score * weights['release_height'] +
        side_score * weights['release_side'] +
        angle_score * weights['horizontal_angle'] +
        speed_diff_score * weights['speed_diff'] +
        h_break_score * weights['horizontal_break'] +
        v_break_score * weights['vertical_break'] +
        distinctive_shape_score * weights['distinctive_shape']
    )
    
    # Convert to Stuff+ scale where 100 = average (0.5 composite score)
    stuff_plus = 100 + (composite_score - 0.5) * 100
    
    # Cap at reasonable bounds
    stuff_plus = max(40, min(160, stuff_plus))
    
    return stuff_plus

def calculate_player_stuff_plus(pitch_data):
    """Calculate Stuff+ for each pitch type for a player"""
    pitch_stuff = {}
    
    # First, calculate the player's fastball velocity
    fastball_data = pitch_data[
        pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)
    ]
    player_fastball_velocity = fastball_data['Velocity'].mean() if len(fastball_data) > 0 else None
    
    # Define pitch type mappings - treat slider and curveball separately
    pitch_type_mapping = {
        'Fastball': ['Fastball'],
        'ChangeUp': ['ChangeUp'],
        'Slider': ['Slider'],
        'Curveball': ['CurveBall'],
        'Cutter': ['Cutter'], 
        'Splitter': ['Splitter']
    }
    
    for pitch_category, pitch_variants in pitch_type_mapping.items():
        # Filter data for this pitch type using exact matching
        pitch_type_data = pitch_data[
            pitch_data['Pitch Type'].isin(pitch_variants)
        ]
        
        if len(pitch_type_data) >= 1:  # Lowered threshold to 1 pitch to show all pitch types
            stuff_plus = calculate_bonnies_stuff_plus_for_pitch_type(
                pitch_type_data, pitch_category, player_fastball_velocity
            )
            
            # Calculate speed differential for display
            avg_velocity = pitch_type_data['Velocity'].mean()
            if pitch_category != 'Fastball' and player_fastball_velocity is not None:
                speed_diff = player_fastball_velocity - avg_velocity
            else:
                speed_diff = 0
            
            pitch_stuff[pitch_category] = {
                'stuff_plus': stuff_plus,
                'count': len(pitch_type_data),
                'avg_velocity': avg_velocity,
                'avg_spin': pitch_type_data['Total Spin'].mean(),
                'avg_h_break': abs(pitch_type_data['HB (trajectory)'].mean()) if 'HB (trajectory)' in pitch_type_data.columns else 0,
                'avg_v_break': pitch_type_data['VB (trajectory)'].mean() if 'VB (trajectory)' in pitch_type_data.columns else 0,
                'speed_diff': speed_diff
            }
    
    return pitch_stuff

def find_available_reports(player_name):
    """Find all available development reports for a player"""
    
    # Format player name to match filename convention
    formatted_name = player_name.replace(" ", "")
    
    # Path to the development reports directory
    reports_dir = os.path.join("data", "Dev Reports")
    
    if not os.path.exists(reports_dir):
        return []
    
    # Look for files matching the pattern: FormattedName*.txt
    pattern = os.path.join(reports_dir, f"{formatted_name}*.txt")
    report_files = glob.glob(pattern)
    
    # Extract dates from filenames and sort
    available_reports = []
    for file_path in report_files:
        filename = os.path.basename(file_path)
        # Extract the date portion (last 6 characters before .txt)
        if filename.endswith('.txt') and len(filename) >= 10:  # Minimum length check
            date_part = filename[-10:-4]  # Gets the 6 digits before .txt
            if date_part.isdigit() and len(date_part) == 6:
                # Format date for display: MMDDYY -> MM/DD/YY
                display_date = f"{date_part[:2]}/{date_part[2:4]}/{date_part[4:]}"
                available_reports.append({
                    'file_path': file_path,
                    'date_code': date_part,
                    'display_date': display_date,
                    'filename': filename
                })
    
    # Sort by date (most recent first)
    available_reports.sort(key=lambda x: x['date_code'], reverse=True)
    
    return available_reports

def load_specific_pitch_development_report(file_path):
    """Load a specific pitch development report from file path"""
    
    try:
        # Try different encodings in case of encoding issues
        encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    report_content = f.read().strip()
                
                if report_content:
                    return report_content
                break
            except UnicodeDecodeError:
                continue
        
        # If we get here, file was found but empty or couldn't be decoded
        return f"Development report found but could not be read properly."
        
    except FileNotFoundError:
        return f"Report file not found: {file_path}"
    except Exception as e:
        return f"Error loading development report: {str(e)}"

def display_pitch_development_report_section(player_name):
    """Display the pitch development report section with date selection"""
    
    st.markdown('<h3 class="section-header">Player Development Report</h3>', unsafe_allow_html=True)
    
    # Find all available reports for this player
    available_reports = find_available_reports(player_name)
    
    if not available_reports:
        # No reports found
        formatted_name = player_name.replace(" ", "")
        st.warning(f"No development reports found for {player_name}")
        st.info(f"Expected filename format: `data/Dev Reports/{formatted_name}MMDDYY.txt` (e.g., {formatted_name}090625.txt)")
        return
    
    # Create date selection buttons or dropdown
    if len(available_reports) == 1:
        # Only one report, just show it with the date
        selected_report = available_reports[0]
        st.info(f"Report Date: {selected_report['display_date']}")
    else:
        # Multiple reports, create selection interface
        st.subheader("Select Report Date:")
        
        # Create columns for date buttons
        cols = st.columns(min(len(available_reports), 4))  # Max 4 columns
        
        # Use session state to track selected report
        if 'selected_report_date' not in st.session_state:
            st.session_state.selected_report_date = available_reports[0]['date_code']  # Default to most recent
        
        selected_report = None
        
        # Create buttons for each available date
        for i, report in enumerate(available_reports):
            col_idx = i % 4
            with cols[col_idx]:
                is_selected = st.session_state.selected_report_date == report['date_code']
                
                # Style the button differently if selected
                if is_selected:
                    button_label = f"🔹 {report['display_date']}"
                else:
                    button_label = report['display_date']
                
                if st.button(button_label, key=f"report_btn_{report['date_code']}"):
                    st.session_state.selected_report_date = report['date_code']
                    st.rerun()
                
                if is_selected:
                    selected_report = report
        
        # If no report is selected (shouldn't happen), default to first
        if selected_report is None:
            selected_report = available_reports[0]
    
    # Display the selected report
    report_content = load_specific_pitch_development_report(selected_report['file_path'])
    
    if report_content.startswith("Development report found") or report_content.startswith("Error loading") or report_content.startswith("Report file not found"):
        # Show error/warning message
        st.error(report_content)
    else:
        # Display the actual report content
        st.markdown(report_content)
        
        # Add metadata about the report
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"*Report Date: {selected_report['display_date']}*")
        with col2:
            pass
        
        # Show total number of reports available
        if len(available_reports) > 1:
            st.caption(f"*{len(available_reports)} reports available for {player_name}*")

def create_stuff_plus_radar_chart(pitch_stuff, player_name):
    """Create radar chart showing Stuff+ for each pitch type using matplotlib"""
    # Use the actual pitch types from the data instead of hardcoded list
    all_pitch_types = list(pitch_stuff.keys())
    
    # If no pitch types, use default
    if not all_pitch_types:
        all_pitch_types = ['Fastball', 'ChangeUp', 'Slider']
    
    # Prepare data for radar chart
    stuff_values = []
    
    for pitch_type in all_pitch_types:
        if pitch_type in pitch_stuff:
            # Convert to percentile (0-1 scale) where 100 = 0.5 (50th percentile)
            stuff_plus_val = pitch_stuff[pitch_type]['stuff_plus']
            # Scale from 40-160 range to 0-1 range with 100 = 0.5
            percentile = (stuff_plus_val - 40) / 120  # 120 is the range (160-40)
            stuff_values.append(percentile)
        else:
            stuff_values.append(0.5)  # Default to 50th percentile if pitch doesn't exist
    
    # Calculate angles for radar chart
    N = len(all_pitch_types)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    # Add first value to end to complete the circle
    stuff_values += stuff_values[:1]
    
    # Reference line at 50th percentile (league average)
    avg_line = [0.5] * len(angles)
    
    # Create the radar chart
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Plot the player's data
    ax.plot(angles, stuff_values, linewidth=3, label=f'{player_name} Stuff+', color='#54342c')
    ax.fill(angles, stuff_values, alpha=0.3, color='#54342c')
    
    # Add reference line at 50th percentile (league average)
    ax.plot(angles, avg_line, linewidth=2, label='League Average (100)', 
           linestyle='--', color='gray', alpha=0.8)
    
    # Customize the chart
    plt.xticks(angles[:-1], all_pitch_types, color='white', size=12)
    plt.yticks(np.linspace(0, 1, 6), ['0%', '20%', '40%', '60%', '80%', '100%'], 
              color='white', size=10)
    plt.title(f'{player_name} Pitch Stuff+ Radar Chart', color='white', size=16, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    return fig

def create_movement_chart(pitch_data):
    """Create movement chart (HB vs VB) using matplotlib"""
    if len(pitch_data) == 0 or 'HB (trajectory)' not in pitch_data.columns or 'VB (trajectory)' not in pitch_data.columns:
        return None
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))  # Make it square for better visualization
    
    # Get unique pitch types and assign colors
    pitch_types = pitch_data['Pitch Type'].unique()
    import numpy
    colors = plt.cm.Set1(numpy.linspace(0, 1, len(pitch_types)))
    
    from matplotlib.patches import Ellipse
    import numpy as np
    
    for i, pitch_type in enumerate(pitch_types):
        pitch_subset = pitch_data[pitch_data['Pitch Type'] == pitch_type]
        
        # Get clean data (remove NaN values)
        hb_data = pitch_subset['HB (trajectory)'].dropna()
        vb_data = pitch_subset['VB (trajectory)'].dropna()
        
        if len(hb_data) > 0 and len(vb_data) > 0:
            # Remove outliers before calculating means and ellipses
            # Use IQR method to identify and remove extreme outliers
            def remove_outliers(data):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.0 * IQR  # More aggressive outlier removal
                upper_bound = Q3 + 2.0 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]
            
            # Only remove outliers if we have enough data points
            if len(hb_data) >= 3:
                hb_clean = remove_outliers(hb_data)
                vb_clean = remove_outliers(vb_data)
                
                # Make sure we still have enough data after cleaning
                if len(hb_clean) >= 1 and len(vb_clean) >= 1:
                    hb_data = hb_clean
                    vb_data = vb_clean
            
            # Calculate means from cleaned data
            mean_hb = hb_data.mean()
            mean_vb = vb_data.mean()
            
            # Plot individual pitches (including outliers for transparency)
            ax.scatter(pitch_subset['HB (trajectory)'], pitch_subset['VB (trajectory)'], 
                      c=[colors[i]], label=pitch_type, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            
            # Add confidence ellipse if we have multiple data points
            if len(hb_data) >= 2 and len(vb_data) >= 2:
                # Calculate standard deviations
                std_hb = hb_data.std()
                std_vb = vb_data.std()
                
                # Use smaller, more realistic ellipse for pitch movement analysis
                # Cap the standard deviations to reasonable limits for pitch movement
                max_std_hb = min(std_hb, 3.0)  # Max 3 inches horizontal variation
                max_std_vb = min(std_vb, 3.0)  # Max 3 inches vertical variation
                
                # Create ellipse representing realistic expected range
                ellipse = Ellipse((mean_hb, mean_vb), 
                                width=2*max_std_hb,  # ~68% of pitches within this range
                                height=2*max_std_vb,
                                facecolor=colors[i], 
                                alpha=0.2, 
                                edgecolor=colors[i], 
                                linewidth=2,
                                linestyle='--')
                ax.add_patch(ellipse)
            
            # Mark the average with a larger, outlined point
            ax.scatter(mean_hb, mean_vb, 
                      c=[colors[i]], s=120, marker='D', 
                      edgecolors='white', linewidth=2, 
                      alpha=0.9, zorder=10)
    
    # Add quadrant lines at (0,0)
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.7, linewidth=1)
    ax.axvline(x=0, color='white', linestyle='-', alpha=0.7, linewidth=1)
    
    # Calculate data range to set symmetric axes centered on (0,0)
    h_break_data = pitch_data['HB (trajectory)'].dropna()
    v_break_data = pitch_data['VB (trajectory)'].dropna()
    
    if len(h_break_data) > 0 and len(v_break_data) > 0:
        # Find the maximum absolute value for both axes to make them symmetric
        max_h = max(abs(h_break_data.min()), abs(h_break_data.max()))
        max_v = max(abs(v_break_data.min()), abs(v_break_data.max()))
        
        # Add some padding (20% extra)
        padding_factor = 1.2
        max_h *= padding_factor
        max_v *= padding_factor
        
        # Set symmetric limits
        ax.set_xlim(-max_h, max_h)
        ax.set_ylim(-max_v, max_v)
    else:
        # Default symmetric limits if no data
        ax.set_xlim(-25, 25)
        ax.set_ylim(-15, 25)
    
    ax.set_xlabel('Horizontal Break (inches)', color='white', fontsize=12)
    ax.set_ylabel('Vertical Break (inches)', color='white', fontsize=12)
    ax.set_title('Pitch Movement Profile', color='white', size=14, weight='bold')
    
    # Add legend with better positioning and explanation
    legend_elements = ax.get_legend_handles_labels()
    legend = ax.legend(*legend_elements, loc='upper right', framealpha=0.8)
    
    # Add explanation text
    ax.text(0.02, 0.02, 'Diamond = Average\nDashed ellipse = Expected range', 
            transform=ax.transAxes, ha='left', va='bottom', 
            color='gray', fontsize=9, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add labels for the quadrants
    ax.text(0.02, 0.98, 'Arm Side\nRising', transform=ax.transAxes, 
            ha='left', va='top', color='gray', fontsize=9, alpha=0.7)
    ax.text(0.98, 0.98, 'Glove Side\nRising', transform=ax.transAxes, 
            ha='right', va='top', color='gray', fontsize=9, alpha=0.7)
    ax.text(0.02, 0.12, 'Arm Side\nSinking', transform=ax.transAxes, 
            ha='left', va='bottom', color='gray', fontsize=9, alpha=0.7)
    ax.text(0.98, 0.12, 'Glove Side\nSinking', transform=ax.transAxes, 
            ha='right', va='bottom', color='gray', fontsize=9, alpha=0.7)
    
    return fig

def create_stuff_plus_bar_chart(pitch_stuff, player_name):
    """Create bar chart showing Stuff+ breakdown by pitch type"""
    if not pitch_stuff:
        return None
    
    # Get all pitch types from the data
    pitch_types = list(pitch_stuff.keys())
    stuff_values = [pitch_stuff[pt]['stuff_plus'] for pt in pitch_types]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))  # Made wider to accommodate more pitch types
    
    bars = ax.bar(pitch_types, stuff_values, color='#54342c', alpha=0.8, edgecolor='white')
    
    # Add value labels on bars
    for bar, value in zip(bars, stuff_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # Add reference line at 100
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='League Average (100)')
    
    ax.set_ylabel('Stuff+', color='white')
    ax.set_xlabel('Pitch Type', color='white')
    ax.set_title(f'{player_name} Stuff+ by Pitch Type', color='white', size=14, weight='bold')
    ax.set_ylim(40, 160)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if there are many pitch types
    if len(pitch_types) > 4:
        plt.xticks(rotation=45, ha='right')
    
    return fig

def display_pitch_stuff_details(pitch_stuff, pitch_type):
    """Display detailed Stuff+ breakdown for a pitch type"""
    if pitch_type not in pitch_stuff:
        st.info(f"No {pitch_type} data available")
        return
    
    data = pitch_stuff[pitch_type]
    
    st.subheader(f"{pitch_type} Stuff+ Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stuff+", f"{data['stuff_plus']:.1f}")
    
    with col2:
        st.metric("Pitch Count", f"{data['count']}")
    
    with col3:
        st.metric("Avg Velocity", f"{data['avg_velocity']:.1f} mph")
    
    with col4:
        st.metric("Avg Spin", f"{data['avg_spin']:.0f} rpm")
    
    # Movement metrics and speed differential
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg H-Break", f"{data['avg_h_break']:.1f} in")
    
    with col2:
        st.metric("Avg V-Break", f"{data['avg_v_break']:.1f} in")
    
    with col3:
        if pitch_type != 'Fastball' and data['speed_diff'] > 0:
            st.metric("Speed Diff vs FB", f"{data['speed_diff']:.1f} mph")
        else:
            st.metric("Speed Diff vs FB", "N/A")
    
    # Stuff+ interpretation
    stuff_plus_val = data['stuff_plus']
    if stuff_plus_val >= 120:
        interpretation = "Elite"
        color = "green"
    elif stuff_plus_val >= 110:
        interpretation = "Above Average"
        color = "blue"
    elif stuff_plus_val >= 90:
        interpretation = "Average"
        color = "gray"
    elif stuff_plus_val >= 80:
        interpretation = "Below Average"
        color = "orange"
    else:
        interpretation = "Poor"
        color = "red"
    
    st.markdown(f"**Stuff+ Grade:** :{color}[{interpretation}]")

# Add these imports to the top of your existing file (after your current imports)
import requests

# VALD API Configuration
VALD_CONFIG = st.secrets["VALD_CONFIG"]

# Add these new functions after your existing functions

@st.cache_data(ttl=300)
def get_access_token():
    """Get access token from VALD API"""
    token_data = {
        "grant_type": "client_credentials",
        "client_id": VALD_CONFIG["client_id"],
        "client_secret": VALD_CONFIG["client_secret"]
    }
    
    response = requests.post(VALD_CONFIG["token_url"], data=token_data)
    return response.json()["access_token"] if response.ok else None

@st.cache_data(ttl=1800)
def fetch_all_vald_profiles():
    """Fetch all profiles from External Profiles API"""
    token = get_access_token()
    if not token:
        return {}
    
    headers = {"Authorization": f"Bearer {token}"}
    profiles_url = f"{VALD_CONFIG['profiles_base_url']}/profiles?tenantId={VALD_CONFIG['tenant_id']}"
    
    try:
        response = requests.get(profiles_url, headers=headers)
        
        if response.ok:
            data = response.json()
            
            if "profiles" in data:
                profiles = data["profiles"]
                profiles_dict = {}
                
                for profile in profiles:
                    profile_id = profile.get('profileId')
                    given_name = profile.get('givenName', '').strip()
                    family_name = profile.get('familyName', '').strip()
                    full_name = f"{given_name} {family_name}".strip()
                    
                    profiles_dict[profile_id] = {
                        'profileId': profile_id,
                        'givenName': given_name,
                        'familyName': family_name,
                        'fullName': full_name,
                        'dateOfBirth': profile.get('dateOfBirth'),
                        'height': profile.get('height'),
                        'weight': profile.get('weight'),
                        'sex': profile.get('sex')
                    }
                
                return profiles_dict
        return {}
    except Exception as e:
        st.error(f"Error fetching profiles: {str(e)}")
        return {}

@st.cache_data(ttl=1800)
def get_vald_team_id():
    """Get team ID from the v2019q3/teams endpoint"""
    token = get_access_token()
    if not token:
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    teams_url = f"{VALD_CONFIG['forcedecks_base_url']}/v2019q3/teams"
    
    try:
        response = requests.get(teams_url, headers=headers)
        
        if response.ok:
            teams = response.json()
            if teams and len(teams) > 0:
                return teams[0].get('id') or teams[0].get('teamId')
        return None
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_player_forcedecks_tests(profile_id, modified_from_date):
    """Fetch ForceDecks test data for a specific player"""
    if not profile_id:
        return pd.DataFrame()
    
    token = get_access_token()
    if not token:
        return pd.DataFrame()
    
    headers = {"Authorization": f"Bearer {token}"}
    modified_date = f"{modified_from_date}T00:00:00.000Z"
    
    initial_url = f"{VALD_CONFIG['forcedecks_base_url']}/tests?tenantId={VALD_CONFIG['tenant_id']}&modifiedFromUtc={modified_date}"
    
    try:
        all_tests = []
        current_url = initial_url
        page_count = 0
        max_pages = 10
        
        while current_url and page_count < max_pages:
            page_count += 1
            response = requests.get(current_url, headers=headers)
            
            if response.status_code == 204:
                break
            
            if response.ok:
                try:
                    data = response.json()
                    tests = data if isinstance(data, list) else data.get("tests", [])
                    
                    if len(tests) > 0:
                        # Filter for the specific player
                        filtered_tests = [test for test in tests if test.get('profileId') == profile_id]
                        all_tests.extend(filtered_tests)
                        
                        if len(tests) > 0:
                            last_test = tests[-1]
                            last_modified = last_test.get('modifiedDateUtc')
                            if last_modified:
                                from datetime import datetime, timedelta
                                last_dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                                next_dt = last_dt + timedelta(microseconds=1)
                                next_modified = next_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z'
                                current_url = f"{VALD_CONFIG['forcedecks_base_url']}/tests?tenantId={VALD_CONFIG['tenant_id']}&modifiedFromUtc={next_modified}"
                            else:
                                current_url = None
                        else:
                            current_url = None
                    else:
                        break
                        
                except Exception as e:
                    st.error(f"Error parsing response: {str(e)}")
                    break
            else:
                break
        
        if all_tests:
            df = pd.DataFrame(all_tests)
            if 'modifiedDateUtc' in df.columns:
                df['modifiedDateUtc'] = pd.to_datetime(df['modifiedDateUtc'], utc=True)
                df['date'] = df['modifiedDateUtc'].dt.date
                df['time'] = df['modifiedDateUtc'].dt.time
            return df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching tests: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_test_trials_for_player(team_id, test_ids):
    """Fetch trials (detailed rep data) for multiple tests"""
    if not team_id or not test_ids:
        return pd.DataFrame()
    
    token = get_access_token()
    if not token:
        return pd.DataFrame()
    
    headers = {"Authorization": f"Bearer {token}"}
    all_trials = []
    
    for test_id in test_ids:
        try:
            trials_url = f"{VALD_CONFIG['forcedecks_base_url']}/v2019q3/teams/{team_id}/tests/{test_id}/trials"
            response = requests.get(trials_url, headers=headers)
            
            if response.ok:
                trials_data = response.json()
                
                if isinstance(trials_data, list) and len(trials_data) > 0:
                    for trial in trials_data:
                        trial['testId'] = test_id
                    all_trials.extend(trials_data)
        
        except Exception:
            continue
    
    if all_trials:
        return pd.DataFrame(all_trials)
    else:
        return pd.DataFrame()

def extract_player_performance_metrics(trials_df, test_data):
    """Extract performance metrics from the results field in trial data"""
    if trials_df.empty:
        return pd.DataFrame()
    
    performance_data = []
    
    # Map trial athleteId to test profileId and testType
    test_mapping = {}
    for _, test in test_data.iterrows():
        test_mapping[test['testId']] = {
            'profileId': test['profileId'],
            'testType': test['testType']
        }
    
    for _, trial in trials_df.iterrows():
        if 'results' not in trial or not trial['results']:
            continue
            
        test_info = test_mapping.get(trial['testId'], {})
        
        for result in trial['results']:
            if not isinstance(result, dict):
                continue
                
            metric_data = {
                'testId': trial['testId'],
                'trialId': trial['id'],
                'athleteId': trial['athleteId'],
                'profileId': test_info.get('profileId', trial['athleteId']),
                'testType': test_info.get('testType', 'Unknown'),
                'recordedUTC': trial['recordedUTC'],
                'resultId': result.get('resultId'),
                'value': result.get('value'),
                'time': result.get('time'),
                'limb': result.get('limb'),
                'repeat': result.get('repeat')
            }
            
            # Extract metric definition
            definition = result.get('definition', {})
            metric_data.update({
                'metric_name': definition.get('name', f"Metric_{result.get('resultId')}"),
                'metric_result': definition.get('result', ''),
                'description': definition.get('description', ''),
                'units': definition.get('unit', ''),
                'repeatable': definition.get('repeatable', False),
                'asymmetry': definition.get('asymmetry', False)
            })
            
            performance_data.append(metric_data)
    
    return pd.DataFrame(performance_data) if performance_data else pd.DataFrame()

def find_player_vald_profile_id(player_name, all_profiles):
    """Find VALD profile ID for a specific player by name matching"""
    for profile_id, profile_data in all_profiles.items():
        if profile_data['fullName'] == player_name:
            return profile_id
    return None

def display_player_force_plate_section(player_name):
    """Display Force Plate section for the selected player"""
    
    st.markdown('<h3 class="section-header">Force Plate Performance Analysis</h3>', unsafe_allow_html=True)
    
    # Initialize VALD profiles in session state if not done
    if 'vald_profiles' not in st.session_state:
        st.session_state.vald_profiles = {}
    if 'vald_profiles_loaded' not in st.session_state:
        st.session_state.vald_profiles_loaded = False
    
    # Load VALD profiles
    if not st.session_state.vald_profiles_loaded:
        with st.spinner("Loading VALD profiles..."):
            st.session_state.vald_profiles = fetch_all_vald_profiles()
            st.session_state.vald_profiles_loaded = True
    
    # Find player's VALD profile ID
    player_profile_id = find_player_vald_profile_id(player_name, st.session_state.vald_profiles)
    
    if not player_profile_id:
        st.warning(f"No VALD profile found for {player_name}")
        
        # Show available profiles for debugging
        with st.expander("Available VALD Profiles (for debugging)", expanded=False):
            profile_names = [profile['fullName'] for profile in st.session_state.vald_profiles.values()]
            st.write(f"Found {len(profile_names)} profiles:")
            for name in sorted(profile_names)[:20]:  # Show first 20
                st.write(f"- {name}")
        return
    
    st.success(f"Found VALD profile for {player_name}")
    
    # Date selection and exercise selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        force_plate_date = st.date_input(
            "Load Force Plate Data From:",
            value=date(2025, 9, 6),
            min_value=date(2025, 9, 1),
            max_value=date.today(),
            key=f"fp_date_{player_name}"
        )

        load_fp_data = st.button("Load Data", type="primary", key=f"load_fp_{player_name}")
    
    with col2:
        # Exercise selection
        exercise_options = {
            "CMJ Performance": "CMJ",
            "Squat Jump Performance": "SJ", 
            "Hop Jump Performance": "HJ",
            "Plyo Pushup Performance": "PPU"
        }
        
        selected_exercise = st.selectbox(
            "Select Exercise Type:",
            options=list(exercise_options.keys()),
            key=f"exercise_select_{player_name}"
        )
        
        exercise_code = exercise_options[selected_exercise]
    
    # Load Force Plate data
    if load_fp_data:
        team_id = get_vald_team_id()
        
        with st.spinner("Loading force plate data..."):
            # Get test summaries for this specific player
            df = fetch_player_forcedecks_tests(player_profile_id, force_plate_date.strftime('%Y-%m-%d'))
            
            if not df.empty:
                st.success(f"Found {len(df)} tests for {player_name}")
                
                # Get trial data
                test_ids = df['testId'].unique().tolist()
                trials_df = fetch_test_trials_for_player(team_id, test_ids)
                
                if not trials_df.empty:
                    # Extract performance metrics
                    perf_df = extract_player_performance_metrics(trials_df, df)
                    
                    if not perf_df.empty:
                        # Store in session state
                        st.session_state[f'fp_data_{player_name}'] = perf_df
                        st.success(f"Loaded {len(perf_df)} performance measurements")
                    else:
                        st.error("No performance metrics extracted from trial data")
                else:
                    st.error("No trial data found")
            else:
                st.warning(f"No test data found for {player_name} from {force_plate_date}")
    
    # Display Force Plate data if available
    if f'fp_data_{player_name}' in st.session_state:
        perf_df = st.session_state[f'fp_data_{player_name}']
        display_selected_exercise_analysis(perf_df, player_name, exercise_code, selected_exercise)

def get_daily_values_for_metric(exercise_data, metric, exercise_code):
    """Get daily values for a metric - all repeats for HJ, max for others"""
    # Filter for the metric and trial limb only
    metric_data = exercise_data[
        (exercise_data['metric_name'] == metric) & 
        (exercise_data['limb'] == 'Trial')
    ].copy()
    
    if metric_data.empty:
        return pd.DataFrame()
    
    if exercise_code == 'HJ':  # Hop Jump - show all repeat numbers
        # For HJ, group by date and include all repeat numbers
        daily_values = []
        for date, group in metric_data.groupby('test_date'):
            for _, row in group.iterrows():
                daily_values.append({
                    'test_date': date, 
                    'value': row['value'],
                    'repeat': row['repeat']
                })
        
        return pd.DataFrame(daily_values) if daily_values else pd.DataFrame()
    else:
        # Other exercises - take max value per day (unchanged)
        daily_max = metric_data.groupby('test_date')['value'].max().reset_index()
        return daily_max.sort_values('test_date')

def create_cmj_quadrant_analysis(exercise_data, player_name):
    """Create a 2x2 quadrant analysis for CMJ data using E:C Ratio methodology with traditional grid"""
    
    # Filter for CMJ data and trial limb only
    cmj_data = exercise_data[
        (exercise_data['testType'] == 'CMJ') & 
        (exercise_data['limb'] == 'Trial')
    ].copy()
    
    if cmj_data.empty:
        return None
    
    # CREATE THE test_date COLUMN HERE (this was missing!)
    cmj_data['test_date'] = pd.to_datetime(cmj_data['recordedUTC']).dt.date
    
    # Define the key metrics we need for E:C Ratio analysis
    # Using the EXACT metric names from VALD API for concentric and eccentric impulse
    required_metrics = {
        'concentric_impulse': ['Concentric Impulse'],
        'eccentric_deceleration': ['Eccentric Deceleration Impulse'],
        'eccentric_braking': ['Eccentric Braking Impulse']
    }
    
    # Extract the metrics we need
    metrics_data = {}
    
    for category, metric_names in required_metrics.items():
        for metric_name in metric_names:
            # Find exact matching metrics in the data
            matching_metrics = cmj_data[cmj_data['metric_name'] == metric_name]
            
            if not matching_metrics.empty:
                # Get daily max values
                daily_values = matching_metrics.groupby('test_date')['value'].max().reset_index()
                
                if len(daily_values) > 0:
                    metrics_data[category] = {
                        'values': daily_values['value'].values,
                        'dates': daily_values['test_date'].values,
                        'latest': daily_values['value'].iloc[-1],
                        'mean': daily_values['value'].mean(),
                        'metric_name': matching_metrics.iloc[0]['metric_name'],
                        'units': matching_metrics.iloc[0].get('units', '')
                    }
                    break  # Use first matching metric found
    
    # Check if we have all required metrics
    if 'concentric_impulse' not in metrics_data:
        st.warning("CMJ quadrant analysis requires Concentric Impulse metric")
        return None
        
    if 'eccentric_deceleration' not in metrics_data or 'eccentric_braking' not in metrics_data:
        st.warning("CMJ quadrant analysis requires both Eccentric Deceleration Impulse and Eccentric Braking Impulse metrics")
        return None
    
    # Get the data
    concentric_data = metrics_data['concentric_impulse']
    eccentric_decel_data = metrics_data['eccentric_deceleration']
    eccentric_brake_data = metrics_data['eccentric_braking']
    
    # Match data points by date for all three metrics
    conc_df = pd.DataFrame({'date': concentric_data['dates'], 'concentric': concentric_data['values']})
    ecc_decel_df = pd.DataFrame({'date': eccentric_decel_data['dates'], 'eccentric_decel': eccentric_decel_data['values']})
    ecc_brake_df = pd.DataFrame({'date': eccentric_brake_data['dates'], 'eccentric_brake': eccentric_brake_data['values']})
    
    # Merge all three metrics on date
    combined_df = pd.merge(conc_df, ecc_decel_df, on='date', how='inner')
    combined_df = pd.merge(combined_df, ecc_brake_df, on='date', how='inner')
    
    if len(combined_df) == 0:
        st.warning("No matching dates between Concentric, Eccentric Deceleration, and Eccentric Braking Impulse data")
        return None
    
    # Calculate combined eccentric impulse (deceleration + braking)
    combined_df['eccentric_total'] = combined_df['eccentric_decel'] + combined_df['eccentric_brake']
    
    # Calculate E:C Ratio using combined eccentric impulse
    combined_df['ec_ratio'] = combined_df['eccentric_total'] / combined_df['concentric']
    
    # Create the quadrant plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Use fixed grid boundaries (not based on data means)
    # Calculate reasonable fixed boundaries based on data range
    x_values = combined_df['concentric'].values
    y_values = combined_df['eccentric_total'].values  # Use combined eccentric values
    
    # Create fixed grid lines at the center of the data range
    x_center = (x_values.min() + x_values.max()) / 2
    y_center = (y_values.min() + y_values.max()) / 2
    
    # Add quadrant grid lines (fixed positioning)
    ax.axhline(y=y_center, color='white', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=x_center, color='white', linestyle='-', alpha=0.8, linewidth=2)
    
    # Plot data points with color and positioning based on E:C ratio classification
    dates = combined_df['date'].values
    ec_ratios = combined_df['ec_ratio'].values
    
    for i, (x_orig, y_orig, date, ec_ratio) in enumerate(zip(x_values, y_values, dates, ec_ratios)):
        
        # Determine classification and quadrant positioning based on E:C ratio
        if ec_ratio < 0.8:
            # Concentric Deficient - place in bottom right quadrant
            color = 'orange'
            classification = 'Concentric Deficient'
            x_plot = x_center + (x_center * 0.3)  # Right side
            y_plot = y_center - (y_center * 0.3)  # Bottom side
        elif ec_ratio > 1.2:
            # Eccentric Deficient - place in top left quadrant  
            color = 'yellow'
            classification = 'Eccentric Deficient'
            x_plot = x_center - (x_center * 0.3)  # Left side
            y_plot = y_center + (y_center * 0.3)  # Top side
        else:
            # Balanced/Maintenance - place in top right quadrant
            color = 'lightgreen'
            classification = 'Balanced'
            x_plot = x_center + (x_center * 0.3)  # Right side
            y_plot = y_center + (y_center * 0.3)  # Top side
        
        # Add some variation so multiple points don't overlap exactly
        if len(x_values) > 1:
            x_offset = (i - len(x_values)/2) * (x_center * 0.05)
            y_offset = (i - len(y_values)/2) * (y_center * 0.05)
            x_plot += x_offset
            y_plot += y_offset
        
        # Plot the point
        ax.scatter(x_plot, y_plot, c=color, s=150, edgecolors='white', linewidth=2, alpha=0.8)
        
        # Add date labels and E:C ratio for all points
        ax.annotate(f'{pd.to_datetime(date).strftime("%m/%d")}\nE:C: {ec_ratio:.2f}', 
                   (x_plot, y_plot), xytext=(10, 10), textcoords='offset points',
                   color='white', fontsize=10, fontweight='bold')
    
    # Set axis limits with better scaling to accommodate all quadrants and data points
    # Calculate ranges and ensure adequate space for all quadrants
    x_range = x_values.max() - x_values.min()
    y_range = y_values.max() - y_values.min()
    
    # Use larger padding to ensure points don't get cut off and quadrants are visible
    x_padding = max(x_range * 0.5, x_center * 0.4)  # At least 50% padding or 40% of center
    y_padding = max(y_range * 0.5, y_center * 0.4)
    
    # Ensure we have space around the center lines for all quadrants
    x_min = min(x_values.min() - x_padding, x_center - x_padding)
    x_max = max(x_values.max() + x_padding, x_center + x_padding)
    y_min = min(y_values.min() - y_padding, y_center - y_padding)
    y_max = max(y_values.max() + y_padding, y_center + y_padding)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add quadrant labels in fixed positions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate positions for quadrant labels
    x_left = xlim[0] + (x_center - xlim[0]) * 0.5
    x_right = x_center + (xlim[1] - x_center) * 0.5
    y_bottom = ylim[0] + (y_center - ylim[0]) * 0.5
    y_top = y_center + (ylim[1] - y_center) * 0.5
    
    # Add quadrant labels with background boxes
    label_style = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
    
    # Top-left: Eccentric Deficient
    ax.text(x_left, y_top, 'ECCENTRIC\nDEFICIENT', 
            ha='center', va='center', color='yellow', fontsize=12, fontweight='bold',
            bbox=label_style)
    
    # Top-right: Maintenance/Balanced
    ax.text(x_right, y_top, 'MAINTENANCE', 
            ha='center', va='center', color='lightgreen', fontsize=12, fontweight='bold',
            bbox=label_style)
    
    # Bottom-left: Foundationally Deficient (if needed for very low values)
    ax.text(x_left, y_bottom, 'LOW OVERALL\nCAPABILITY', 
            ha='center', va='center', color='lightcoral', fontsize=12, fontweight='bold',
            bbox=label_style)
    
    # Bottom-right: Concentric Deficient
    ax.text(x_right, y_bottom, 'CONCENTRIC\nDEFICIENT', 
            ha='center', va='center', color='orange', fontsize=12, fontweight='bold',
            bbox=label_style)
    
    # Customize the plot
    ax.set_xlabel(f"Concentric Impulse ({concentric_data['units']})", color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel(f"Combined Eccentric Impulse ({eccentric_decel_data['units']})", color='white', fontsize=14, fontweight='bold')
    ax.set_title(f'{player_name} - CMJ E:C Ratio Analysis', 
                color='white', fontsize=16, fontweight='bold', pad=20)
    
    # Style the axes
    ax.tick_params(axis='both', colors='white', labelsize=12)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white', linestyle=':', linewidth=0.5)
    
    return fig

def display_selected_exercise_analysis(perf_df, player_name, exercise_code, exercise_display_name):
    """Display analysis for the selected exercise with daily max values for progression"""
    
    if perf_df.empty:
        st.warning(f"No force plate data available for {player_name}")
        return
    
    # Clean up test types - fix SLJ to SJ
    perf_df['testType'] = perf_df['testType'].replace('SLJ', 'SJ')
    
    # Filter for the selected exercise
    exercise_data = perf_df[perf_df['testType'] == exercise_code].copy()
    
    if exercise_data.empty:
        st.warning(f"No {exercise_display_name} data found for {player_name}")
        available_tests = perf_df['testType'].unique()
        st.info(f"Available test types: {list(available_tests)}")
        return
    
    st.subheader(f"{exercise_display_name} Analysis")
    
    # Key metrics for each test type - use exact names that have all repeats
    key_metrics = {
        'CMJ': ['Jump Height (Flight Time)', 'Peak Power', 'Peak Force', 'RSI-modified'],
        'SJ': ['Jump Height (Flight Time)', 'Peak Power', 'Peak Force', 'Takeoff Peak Force'],
        'PPU': ['Peak Power', 'Peak Force', 'Flight Time'],
        'HJ': ['Jump Height (Flight Time)', 'Peak Force', 'Landing RFD', 'Time to Peak Force']
    }
    
    available_metrics = exercise_data['metric_name'].unique()
    target_metrics = key_metrics.get(exercise_code, [])
    
    # Find metrics that match our targets (partial matching)
    matched_metrics = []
    for target in target_metrics:
        for available in available_metrics:
            if target.lower() in available.lower():
                matched_metrics.append(available)
                break
    
    # Special handling for CMJ - show quadrant analysis instead of progression charts
    if exercise_code == 'CMJ':
    
        quadrant_chart = create_cmj_quadrant_analysis(exercise_data, player_name)
        if quadrant_chart:
            st.pyplot(quadrant_chart)
            plt.close()
            
            # Add interpretation text below the graph
            st.markdown("### Quadrant Interpretation:")
            st.markdown("""
            - **Maintenance**: Good strength profile, focus on skill refinement
            - **Concentric Deficient**: Develop explosive power and rate of force development  
            - **Eccentric Deficient**: Build eccentric strength and force absorption capacity
            - **Foundationally Deficient**: Build fundamental strength in both phases
       
            """)
    else:
        pass
    
    # Add exact matches for HJ metrics that might have different names
    if exercise_code == 'HJ':
        for available in available_metrics:
            if 'jump height' in available.lower() and available not in matched_metrics:
                matched_metrics.append(available)
            elif 'peak force' in available.lower() and available not in matched_metrics:
                matched_metrics.append(available)
            elif 'landing rfd' in available.lower() and available not in matched_metrics:
                matched_metrics.append(available)
        
        # Filter out metrics with "Best", "Mean", or "Fatigue" in the name for HJ
        matched_metrics = [metric for metric in matched_metrics 
                          if 'best' not in metric.lower() and 'mean' not in metric.lower() and 'fatigue' not in metric.lower()]
    
    if not matched_metrics:
        st.warning(f"No key metrics found for {exercise_display_name}")
        st.write(f"Available metrics: {list(available_metrics)[:5]}...")
        return
    
    # Convert recordedUTC to date for daily grouping
    exercise_data['test_date'] = pd.to_datetime(exercise_data['recordedUTC']).dt.date
    
    # Overall summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_test_days = len(exercise_data['test_date'].unique())
        st.metric("Test Days", total_test_days)
    
    with col2:
        if exercise_code == 'HJ':
            # For HJ, show number of repeats
            total_repeats = len(exercise_data[exercise_data['limb'] == 'Trial']['repeat'].unique())
            st.metric("Total Repeats", total_repeats)
        else:
            # For other exercises, show trial count
            total_trials = len(exercise_data['trialId'].unique())
            st.metric("Total Tests", total_trials)
    
    with col3:
        total_reps = len(exercise_data)
        st.metric("Total Reps", total_reps)
    
    with col4:
        latest_date = exercise_data['test_date'].max()
        if pd.notna(latest_date):
            latest_date_str = latest_date.strftime('%m/%d/%y')
            st.metric("Latest Test", latest_date_str)
        else:
            st.metric("Latest Test", "N/A")
    
    # For HJ, display fatigue metrics separately (not as graphs)
    if exercise_code == 'HJ':
        fatigue_metrics = [metric for metric in available_metrics if 'fatigue' in metric.lower()]
        
        # Filter out specific metrics we don't want to display
        excluded_terms = ['hops/reps', 'peak power']
        filtered_fatigue_metrics = []
        
        for metric in fatigue_metrics:
            should_include = True
            for excluded_term in excluded_terms:
                if excluded_term in metric.lower():
                    should_include = False
                    break
            if should_include:
                filtered_fatigue_metrics.append(metric)
        
        if filtered_fatigue_metrics:
            st.subheader("Fatigue Analysis")
            fatigue_cols = st.columns(min(len(filtered_fatigue_metrics), 4))
            
            for i, metric in enumerate(filtered_fatigue_metrics):
                col_idx = i % 4
                
                # Get fatigue metric data
                fatigue_data = exercise_data[
                    (exercise_data['metric_name'] == metric) & 
                    (exercise_data['limb'] == 'Trial')
                ].copy()
                
                if not fatigue_data.empty:
                    units = fatigue_data.iloc[0].get('units', '')
                    latest_value = fatigue_data.iloc[-1]['value'] if len(fatigue_data) > 0 else 0
                    avg_value = fatigue_data['value'].mean()
                    
                    with fatigue_cols[col_idx]:
                        st.metric(
                            f"{metric.replace('Fatigue', '').strip()}",
                            f"{latest_value:.2f} {units}",
                            f"Avg: {avg_value:.2f}"
                        )
    
    # Display key metrics in columns - SKIP FOR HJ
    if exercise_code != 'HJ':
        st.subheader("Current Performance Metrics")
        cols = st.columns(min(len(matched_metrics), 4))
        
        for i, metric in enumerate(matched_metrics[:4]):  # Show up to 4 metrics
            col_idx = i % 4
            
            # Filter for the metric and trial limb only
            metric_data = exercise_data[
                (exercise_data['metric_name'] == metric) & 
                (exercise_data['limb'] == 'Trial')
            ].copy()
            
            if not metric_data.empty:
                # Get units and best value
                units = metric_data.iloc[0].get('units', '')
                best_value = metric_data['value'].max()
                recent_value = metric_data.iloc[-1]['value'] if len(metric_data) > 0 else best_value
                
                with cols[col_idx]:
                    st.metric(
                        f"{metric.replace('(Flight Time)', '')}",
                        f"{best_value:.2f} {units}",
                        f"Recent: {recent_value:.2f}"
                    )
    
    # Show progression charts for each key metric
    if exercise_code == 'HJ':
        st.subheader("Performance by Repeat Number")
    else:
        st.subheader("Performance Progression (Daily Max Values)")
    
    for metric in matched_metrics:
        # Get data based on exercise type
        daily_data = get_daily_values_for_metric(exercise_data, metric, exercise_code)
        
        if daily_data.empty:
            continue
        
        # Get units for labeling
        metric_sample = exercise_data[
            (exercise_data['metric_name'] == metric) & 
            (exercise_data['limb'] == 'Trial')
        ].iloc[0] if not exercise_data[
            (exercise_data['metric_name'] == metric) & 
            (exercise_data['limb'] == 'Trial')
        ].empty else None
        
        units = metric_sample.get('units', '') if metric_sample is not None else ''
        
        # Create chart
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        if exercise_code == 'HJ':
            # For Hop Jump, get data directly from exercise_data for THIS SPECIFIC METRIC
            metric_specific_data = exercise_data[
                (exercise_data['metric_name'] == metric) & 
                (exercise_data['limb'] == 'Trial')
            ].copy()
            
            if not metric_specific_data.empty:
                repeats = metric_specific_data['repeat'].values
                values = metric_specific_data['value'].values
                
                # Sort by repeat number for proper display
                sorted_indices = np.argsort(repeats)
                repeats = repeats[sorted_indices]
                values = values[sorted_indices]
                
                # Plot each repeat
                ax.scatter(repeats, values, 
                          s=150, color='#54342c', edgecolors='white', linewidth=3, alpha=0.8)
                
                # Connect points with line
                ax.plot(repeats, values, color='#54342c', linewidth=2, alpha=0.7)
                
                # Add value annotations
                for repeat_num, value in zip(repeats, values):
                    ax.annotate(f'{value:.1f}', 
                               (repeat_num, value),
                               textcoords="offset points", xytext=(0,15), ha='center',
                               color='white', fontsize=11, fontweight='bold')
                
                # Add average line for this metric
                avg_value = np.mean(values)
                ax.axhline(y=avg_value, color='white', linestyle='--', alpha=0.7, linewidth=2, label=f'Average: {avg_value:.1f}')
                
                ax.set_xlabel('Repeat Number', color='white', fontsize=14)
                ax.set_title(f'{metric} - All Repeats', color='white', fontweight='bold', fontsize=16, pad=20)
                ax.set_xticks(repeats)
                ax.set_xticklabels([f'R{int(r)}' for r in repeats])
            
        else:
            # For other exercises, show daily max progression (unchanged)
            if len(daily_data) >= 2:
                ax.plot(daily_data['test_date'], daily_data['value'], 
                       marker='o', linewidth=3, markersize=10, color='#54342c', 
                       markerfacecolor='#54342c', markeredgecolor='white', markeredgewidth=2)
                
                # Add trend line if enough data points
                if len(daily_data) >= 3:
                    date_numeric = pd.to_datetime(daily_data['test_date']).map(pd.Timestamp.toordinal)
                    z = np.polyfit(date_numeric, daily_data['value'], 1)
                    p = np.poly1d(z)
                    trend_values = p(date_numeric)
                    
                    ax.plot(daily_data['test_date'], trend_values, 
                           "--", alpha=0.9, color='white', linewidth=2, label='Trend Line')
                    
                    trend_change = ((trend_values.iloc[-1] - trend_values.iloc[0]) / trend_values.iloc[0]) * 100
                    trend_text = f"Trend: {trend_change:+.1f}%"
                    
                    ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#54342c', alpha=0.8, edgecolor='white'),
                            color='white', fontsize=12, fontweight='bold')
                
                ax.set_xlabel('Date', color='white', fontsize=14)
                ax.set_title(f'{metric} - Daily Max Progression', color='white', fontweight='bold', fontsize=16, pad=20)
                ax.tick_params(axis='x', rotation=45)
            else:
                # Single data point
                test_date = daily_data.iloc[0]['test_date']
                value = daily_data.iloc[0]['value']
                ax.scatter([test_date], [value], 
                          s=200, color='#54342c', edgecolors='white', linewidth=3, zorder=5)
                ax.annotate(f'{value:.2f}', 
                           (test_date, value),
                           textcoords="offset points", xytext=(0,25), ha='center',
                           color='white', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', color='white', fontsize=14)
                ax.set_title(f'{metric} - Baseline Performance', color='white', fontweight='bold', fontsize=16, pad=20)
        
        ax.set_ylabel(f'{metric} ({units})', color='white', fontsize=14)
        ax.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        
        if exercise_code == 'HJ' or (exercise_code != 'HJ' and len(daily_data) >= 3):
            legend = ax.legend(loc='upper right', framealpha=0.9)
            legend.get_frame().set_facecolor('#1e1e1e')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show improvement statistics for each metric - SKIP FOR HJ
        if exercise_code != 'HJ':
            # For other exercises, show improvement if multiple data points
            if len(daily_data) >= 2:
                first_value = daily_data.iloc[0]['value']
                last_value = daily_data.iloc[-1]['value']
                improvement = ((last_value - first_value) / first_value) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Starting Best", f"{first_value:.2f} {units}")
                with col2:
                    st.metric("Current Best", f"{last_value:.2f} {units}")
                with col3:
                    st.metric("Improvement", f"{improvement:+.1f}%")
        
        st.markdown("---")
    
    # Show recent test history table - Use existing data, no API call
    st.subheader("Recent Test History")
    
    # Use the existing exercise_data that was already filtered for this exercise type
    recent_trial_data = exercise_data[exercise_data['limb'] == 'Trial'].copy()
    
    if not recent_trial_data.empty:
        # Sort by most recent
        recent_trial_data = recent_trial_data.sort_values('recordedUTC', ascending=False)
        
        if exercise_code == 'HJ':
            # For HJ, group by trialId and show best/mean values
            unique_trials = recent_trial_data['trialId'].unique()[:10]  # Show last 10 trials
            table_data = []
            
            for trial_id in unique_trials:
                trial_metrics = recent_trial_data[recent_trial_data['trialId'] == trial_id]
                
                if trial_metrics.empty:
                    continue
                
                first_row = trial_metrics.iloc[0]
                test_datetime = pd.to_datetime(first_row['recordedUTC'])
                
                row_data = {
                    'Date': test_datetime.strftime('%m/%d/%y'),
                    'Trial ID': trial_id
                }
                
                # Get all unique metrics for this trial
                trial_metric_names = trial_metrics['metric_name'].unique()
                
                # Calculate best and mean values for each metric
                for metric_name in trial_metric_names:
                    metric_data = trial_metrics[trial_metrics['metric_name'] == metric_name]
                    if not metric_data.empty:
                        values = metric_data['value'].values
                        units = metric_data.iloc[0].get('units', '')
                        
                        best_value = values.max()
                        mean_value = values.mean()
                        
                        # Clean up metric name for display
                        clean_metric = metric_name.replace('(Flight Time)', '').strip()
                        
                        row_data[f'{clean_metric} (Best)'] = f"{best_value:.2f} {units}".strip()
                        row_data[f'{clean_metric} (Mean)'] = f"{mean_value:.2f} {units}".strip()
                
                table_data.append(row_data)
        else:
            # For other exercises, show trial-based data
            unique_trials = recent_trial_data['trialId'].unique()[:20]
            table_data = []
            
            for trial_id in unique_trials:
                trial_metrics = recent_trial_data[recent_trial_data['trialId'] == trial_id]
                
                if trial_metrics.empty:
                    continue
                
                first_row = trial_metrics.iloc[0]
                test_datetime = pd.to_datetime(first_row['recordedUTC'])
                
                row_data = {
                    'Date': test_datetime.strftime('%m/%d/%y'),
                    'Trial ID': trial_id
                }
                
                target_metrics = {
                    'Flight Time': 'Flight Time',
                    'Peak Power / BM': 'Peak Power / BM', 
                    'Takeoff Peak Force': 'Takeoff Peak Force',
                    'RSI-modified': 'RSI-modified'
                }
                
                for display_name, metric_name in target_metrics.items():
                    metric_data = trial_metrics[
                        trial_metrics['metric_name'].str.contains(metric_name, case=False, na=False)
                    ]
                    
                    if not metric_data.empty:
                        value = metric_data.iloc[0]['value']
                        units = metric_data.iloc[0].get('units', '')
                        row_data[display_name] = f"{value:.2f} {units}".strip()
                    else:
                        row_data[display_name] = "N/A"
                
                table_data.append(row_data)
        
        if table_data:
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, hide_index=True, use_container_width=True)
        else:
            st.info("No trial data available")
    else:
        st.info("No trial data available for this exercise type")

def biomechanics_display(player_name):
    """Display biomechanical chart for the selected player."""
    st.subheader("Kinematics Sequence & Key Metrics Chart Display:")
    
    # Format player name to match filename convention
    formatted_name = player_name.replace(" ", "")
    
    # Path to the biomech reports directory
    biomech_dir = os.path.join("data", "Biomech Reports")
    
    if not os.path.exists(biomech_dir):
        st.error("Biomechanics Reports directory not found")
        return
    
    # Look for image files with common extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    available_images = []
    
    for ext in image_extensions:
        pattern = os.path.join(biomech_dir, f"{formatted_name}*.{ext}")
        image_files = glob.glob(pattern)
        
        for file_path in image_files:
            filename = os.path.basename(file_path)
            name_without_ext = filename.rsplit('.', 1)[0]
            
            # Extract date if it follows MMDDYY pattern
            if len(name_without_ext) >= 6:
                date_part = name_without_ext[-6:]
                if date_part.isdigit():
                    display_date = f"{date_part[:2]}/{date_part[2:4]}/{date_part[4:]}"
                    available_images.append({
                        'file_path': file_path,
                        'date_code': date_part,
                        'display_date': display_date,
                        'filename': filename
                    })
    
    if available_images:
        # Sort by date (most recent first)
        available_images.sort(key=lambda x: x['date_code'], reverse=True)
        
        # Use the most recent image
        latest_image = available_images[0]
        
        st.image(latest_image['file_path'], 
                caption=f"Biomechanical Analysis for {player_name} - {latest_image['display_date']}")
    else:
        st.warning(f"No biomechanical chart found for {player_name}")

def display_player_assessment_data(player_name):
    """Display the specific player's assessment data with outlier highlighting."""
    st.markdown('<h3 class="section-header">Player Assessment Data</h3>', unsafe_allow_html=True)
    
    excel_file_path = os.path.join("data", "BonniesBaseballTableAssessment.xlsx")
    
    if not os.path.exists(excel_file_path):
        st.error("Assessment table file not found at data/BonniesBaseballTableAssessment.xlsx")
        return
    
    try:
        # Read the Excel file - get all columns now (A through AC)
        df = pd.read_excel(excel_file_path, engine='openpyxl')
        
        # Clean up the dataframe
        df = df.dropna(how='all')
        
        # Split the selected player name to get first and last name
        name_parts = player_name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
        else:
            st.warning(f"Could not parse player name: {player_name}")
            return
        
        # Find the player's row
        player_row = None
        
        if 'First Name' in df.columns and 'Last Name' in df.columns:
            exact_match = df[
                (df['First Name'].str.strip().str.lower() == first_name.lower()) & 
                (df['Last Name'].str.strip().str.lower() == last_name.lower())
            ]
            
            if not exact_match.empty:
                player_row = exact_match.iloc[0]
            else:
                partial_match = df[
                    (df['First Name'].str.contains(first_name, case=False, na=False)) & 
                    (df['Last Name'].str.contains(last_name, case=False, na=False))
                ]
                
                if not partial_match.empty:
                    player_row = partial_match.iloc[0]
                    st.info(f"Found partial match for {player_name}")
        else:
            st.error("Could not find 'First Name' and 'Last Name' columns in the assessment table")
            return
        
        if player_row is not None:
            st.success(f"Found assessment data for {player_name}")
            
            # Calculate statistics for numeric columns (for the entire team)
            assessment_columns = df.columns[:22] if len(df.columns) >= 22 else df.columns
            stats_info = {}
            
            for column in assessment_columns:
                numeric_col = pd.to_numeric(df[column], errors='coerce')
                
                if numeric_col.notna().sum() >= 3:
                    median = numeric_col.median()
                    std = numeric_col.std()
                    
                    if std > 0:
                        stats_info[column] = {
                            'median': median,
                            'std': std,
                            'lower_bound': median - 2 * std,
                            'upper_bound': median + 2 * std
                        }
            
            # Get player's assessment data
            assessment_data = player_row[assessment_columns]
            
            # Display legend
            st.markdown("""
            <div style='padding: 10px; background-color: #000000; border-radius: 5px; margin-bottom: 10px; color: white;'>
                <strong>Legend:</strong> 
                <span style='background-color: #cc0000; color: white; padding: 2px 8px; margin: 0 5px; border-radius: 3px; font-weight: bold;'>Mobility Deficiency (< Median - 2 SD)</span>
                <span style='background-color: #006400; color: white; padding: 2px 8px; margin: 0 5px; border-radius: 3px; font-weight: bold;'>Hyper Mobility (> Median + 2 SD)</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display key metrics in a grid with color coding
            st.subheader("Key Assessment Metrics")
            
            skip_columns = ['First Name', 'Last Name', 'Name', 'Player', 'ID']
            
            # Create metrics with highlighting
            cols = st.columns(4)
            col_idx = 0
            
            for column, value in assessment_data.items():
                if column not in skip_columns and pd.notna(value) and str(value).strip() != '':
                    # Check if this is a numeric column with stats
                    is_outlier = False
                    outlier_type = None
                    
                    if column in stats_info:
                        try:
                            numeric_value = float(value)
                            if numeric_value < stats_info[column]['lower_bound']:
                                is_outlier = True
                                outlier_type = 'low'
                            elif numeric_value > stats_info[column]['upper_bound']:
                                is_outlier = True
                                outlier_type = 'high'
                        except (ValueError, TypeError):
                            pass
                    
                    with cols[col_idx % 4]:
                        if is_outlier:
                            if outlier_type == 'low':
                                st.markdown(f"""
                                <div style='background-color: #cc0000; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;'>
                                    <div style='font-size: 0.8em; margin-bottom: 5px;'>{column}</div>
                                    <div style='font-size: 1.5em;'>{value}</div>
                                    <div style='font-size: 0.7em; margin-top: 5px;'>Mobility Deficiency</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:  # high
                                st.markdown(f"""
                                <div style='background-color: #006400; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;'>
                                    <div style='font-size: 0.8em; margin-bottom: 5px;'>{column}</div>
                                    <div style='font-size: 1.5em;'>{value}</div>
                                    <div style='font-size: 0.7em; margin-top: 5px;'>Hyper Mobility</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.metric(column, str(value))
                    
                    col_idx += 1
                    
                    if col_idx % 4 == 0:
                        cols = st.columns(4)
            
            # Training Translation Section (only if outliers exist)
            outliers_exist = False
            for column, value in assessment_data.items():
                if column in stats_info and pd.notna(value):
                    try:
                        numeric_value = float(value)
                        if numeric_value < stats_info[column]['lower_bound'] or numeric_value > stats_info[column]['upper_bound']:
                            outliers_exist = True
                            break
                    except (ValueError, TypeError):
                        pass
            
            if outliers_exist:
                st.markdown("---")
                st.markdown('<h3 class="section-header">Training Translation Guide</h3>', unsafe_allow_html=True)
                
                st.markdown("""
                **Mobility Deficiency (Low Values):**
                
                Player shows restricted range of motion. Focus on mobility work, dynamic stretching, and tissue quality exercises to improve movement capacity. Prioritize addressing movement restrictions before adding load or intensity.
                
                - Increase mobility drills and dynamic warm-ups
                - Address tissue quality (foam rolling, soft tissue work)
                - Focus on controlled articular rotations (CARs)
                - Gradually expand range of motion through progressive stretching
                """)
                
                st.markdown("""
                **Hyper Mobility (High Values):**
                
                Player demonstrates excellent range of motion. Ensure adequate strength development to support and control this mobility, particularly at end ranges of motion.
                
                - Emphasize strength training throughout full range of motion
                - Focus on eccentric control and end-range strength
                - Implement tempo work and isometric holds
                - Develop motor control to utilize available mobility effectively
                """)
            
            # Display comments section (columns W onwards)
            if len(df.columns) > 22:
                st.markdown("---")
                st.subheader("Player Comments & Notes")
                
                comment_columns = df.columns[22:]
                comment_data = player_row[comment_columns]
                
                has_comments = False
                
                for column, value in comment_data.items():
                    if pd.notna(value) and str(value).strip() != '':
                        has_comments = True
                        
                        if pd.isna(column) or str(column).startswith('Unnamed:') or str(column).strip() == '':
                            display_name = "Further Comments"
                        else:
                            display_name = str(column)
                        
                        with st.expander(f"{display_name}", expanded=True):
                            st.write(str(value))
                
                if not has_comments:
                    st.info("No additional comments or notes available for this player.")
        
        else:
            st.warning(f"No assessment data found for {player_name}")
            st.info(f"Searched for: First Name = '{first_name}', Last Name = '{last_name}'")
    
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        st.info("Please ensure the file is a valid Excel file (.xlsx format)")
        
# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">St. Bonaventure Baseball</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"> Total Player Profile Display </p>', unsafe_allow_html=True)
    
    # Load all player data
    try:
        all_players = load_all_player_data()
    except Exception as e:
        st.error(f"Failed to load player data: {str(e)}")
        st.stop()
    
    if not all_players:
        st.error("No player data found. Please check your data directory.")
        st.stop()
    
    # Sidebar with logo
    try:
        st.sidebar.image("images/liquid_logo.png", width=250)
        st.sidebar.image("images/logo.png", width=250)
    except FileNotFoundError:
        st.sidebar.warning("Logo not found at images/logo.png")
    
    st.sidebar.title("Player Selection")
    st.sidebar.markdown("---")
    
    # Player selector
    player_names = sorted(all_players.keys())
    selected_player = st.sidebar.selectbox("Select Player", player_names)
    
    if selected_player:
        player_data = all_players[selected_player]
        pitch_data = player_data['pitch_data']
        handedness = player_data['handedness']
        
        # Player header
        st.markdown(f'<h2 class="player-header">{selected_player} ({handedness})</h2>', unsafe_allow_html=True)
        
        # Calculate Stuff+ for each pitch type
        pitch_stuff = calculate_player_stuff_plus(pitch_data)
        
        if not pitch_stuff:
            st.warning("No pitch types with sufficient data (minimum 3 pitches) for Stuff+ calculation.")
            st.stop()
        
        # Overall Stuff+ Summary
        st.markdown('<h3 class="section-header">Bonnies Stuff+ Overview</h3>', unsafe_allow_html=True)
        
        # Calculate overall stuff+
        overall_stuff_plus = np.mean([data['stuff_plus'] for data in pitch_stuff.values()])
        total_pitches = sum([data['count'] for data in pitch_stuff.values()])
        pitch_types_count = len(pitch_stuff)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Stuff+", f"{overall_stuff_plus:.1f}")
        
        with col2:
            st.metric("Pitch Types", f"{pitch_types_count}")
        
        with col3:
            st.metric("Total Pitches", f"{total_pitches:,}")
        
        with col4:
            best_pitch = max(pitch_stuff.keys(), key=lambda x: pitch_stuff[x]['stuff_plus'])
            st.metric("Best Pitch", f"{best_pitch}")
    
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">Stuff+ Radar Chart</h3>', unsafe_allow_html=True)
            radar_chart = create_stuff_plus_radar_chart(pitch_stuff, selected_player)
            if radar_chart:
                st.pyplot(radar_chart)
        
        with col2:
            st.markdown('<h3 class="section-header">Stuff+ by Pitch Type</h3>', unsafe_allow_html=True)
            bar_chart = create_stuff_plus_bar_chart(pitch_stuff, selected_player)
            if bar_chart:
                st.pyplot(bar_chart)
        
        # Detailed pitch analysis tabs
        st.markdown('<h3 class="section-header">Detailed Pitch Analysis</h3>', unsafe_allow_html=True)
        
        # Create tabs for each pitch type
        pitch_types = list(pitch_stuff.keys())
        
        if len(pitch_types) > 0:
            tabs = st.tabs(pitch_types)
            
            for i, pitch_type in enumerate(pitch_types):
                with tabs[i]:
                    display_pitch_stuff_details(pitch_stuff, pitch_type)
        
        # Movement analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">Movement Profile</h3>', unsafe_allow_html=True)
            movement_chart = create_movement_chart(pitch_data)
            if movement_chart:
                st.pyplot(movement_chart)
            else:
                st.info("Movement data not available")
        
        with col2:
            # Use the new manual report function instead of auto-generation
            display_pitch_development_report_section(selected_player)
        
        # Stuff+ Summary Table
        st.markdown('<h3 class="section-header">Bonnies Stuff+ Summary</h3>', unsafe_allow_html=True)
        
        # Create summary dataframe with ALL pitch types from the data
        summary_data = []
        
        # Get all unique pitch types from the actual data, excluding invalid entries
        all_pitch_types_in_data = pitch_data['Pitch Type'].unique()
        valid_pitch_types = [pt for pt in all_pitch_types_in_data if pt not in ['-', '', None] and pd.notna(pt)]
        
        for pitch_type in valid_pitch_types:
            # Get data for this pitch type
            pitch_subset = pitch_data[pitch_data['Pitch Type'] == pitch_type]
            
            if len(pitch_subset) > 0:
                # Apply same outlier filtering logic as movement chart
                hb_data = pitch_subset['HB (trajectory)'].dropna()
                vb_data = pitch_subset['VB (trajectory)'].dropna()
                velocity_data = pitch_subset['Velocity'].dropna()
                spin_data = pitch_subset['Total Spin'].dropna()
                
                # Convert to numeric and drop any remaining non-numeric values
                hb_data = pd.to_numeric(hb_data, errors='coerce').dropna()
                vb_data = pd.to_numeric(vb_data, errors='coerce').dropna()
                velocity_data = pd.to_numeric(velocity_data, errors='coerce').dropna()
                spin_data = pd.to_numeric(spin_data, errors='coerce').dropna()
                
                # Remove outliers if we have enough data points
                def remove_outliers_for_summary(data):
                    if len(data) >= 3:
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:  # Avoid division issues
                            lower_bound = Q1 - 2.0 * IQR
                            upper_bound = Q3 + 2.0 * IQR
                            filtered = data[(data >= lower_bound) & (data <= upper_bound)]
                            return filtered if len(filtered) >= 1 else data
                    return data
                
                # Apply outlier filtering to each metric
                hb_clean = remove_outliers_for_summary(hb_data)
                vb_clean = remove_outliers_for_summary(vb_data)
                velocity_clean = remove_outliers_for_summary(velocity_data)
                spin_clean = remove_outliers_for_summary(spin_data)
                
                # Calculate speed differential
                if pitch_type != 'Fastball':
                    fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                    if len(fastball_data) > 0:
                        fb_velo_data = pd.to_numeric(fastball_data['Velocity'], errors='coerce').dropna()
                        fb_velo = fb_velo_data.mean() if len(fb_velo_data) > 0 else 0
                        pitch_velo = velocity_clean.mean() if len(velocity_clean) > 0 else 0
                        speed_diff = fb_velo - pitch_velo if fb_velo > 0 and pitch_velo > 0 else 0
                    else:
                        speed_diff = 0
                else:
                    speed_diff = 0
                
                # Get Stuff+ if available, otherwise calculate it directly
                stuff_plus_val = pitch_stuff.get(pitch_type, {}).get('stuff_plus', None)
                
                # If not in pitch_stuff, calculate it directly for this pitch type
                if stuff_plus_val is None and len(velocity_clean) > 0:
                    # Calculate fastball velocity for speed differential
                    fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                    player_fastball_velocity = None
                    if len(fastball_data) > 0:
                        fb_velo_data = pd.to_numeric(fastball_data['Velocity'], errors='coerce').dropna()
                        player_fastball_velocity = fb_velo_data.mean() if len(fb_velo_data) > 0 else None
                    
                    # Calculate Stuff+ directly for this pitch type
                    try:
                        stuff_plus_val = calculate_bonnies_stuff_plus_for_pitch_type(
                            pitch_subset, pitch_type, player_fastball_velocity
                        )
                    except:
                        stuff_plus_val = None
                
                summary_data.append({
                    'Pitch Type': pitch_type,
                    'Stuff+': stuff_plus_val if stuff_plus_val is not None else 'N/A',
                    'Count': len(pitch_subset),
                    'Avg Velocity': round(velocity_clean.mean(), 1) if len(velocity_clean) > 0 else 0,
                    'Avg Spin': round(spin_clean.mean(), 0) if len(spin_clean) > 0 else 0,
                    'H-Break': round(abs(hb_clean.mean()), 1) if len(hb_clean) > 0 else 0,
                    'V-Break': round(vb_clean.mean(), 1) if len(vb_clean) > 0 else 0,
                    'Speed Diff': round(speed_diff, 1)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Handle mixed data types in Stuff+ column for sorting
        def sort_stuff_plus(df):
            # Create a numeric version for sorting, putting N/A values at the end
            df_copy = df.copy()
            df_copy['stuff_plus_numeric'] = pd.to_numeric(df_copy['Stuff+'], errors='coerce')
            # Sort by numeric values (NaN values go to end automatically)
            df_sorted = df_copy.sort_values('stuff_plus_numeric', ascending=False, na_position='last')
            # Drop the helper column
            return df_sorted.drop('stuff_plus_numeric', axis=1)
        
        summary_df = sort_stuff_plus(summary_df)
        
        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Stuff+": st.column_config.NumberColumn("Stuff+", format="%.1f"),
                "Count": st.column_config.NumberColumn("Count", format="%.0f"),
                "Avg Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
                "Avg Spin": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
                "H-Break": st.column_config.NumberColumn("H-Break", format="%.1f in"),
                "V-Break": st.column_config.NumberColumn("V-Break", format="%.1f in"),
                "Speed Diff": st.column_config.NumberColumn("Speed Diff vs FB", format="%.1f mph")
            }
        )
        
        # Recent pitch data section
        st.markdown('<h3 class="section-header">Recent Pitch Data</h3>', unsafe_allow_html=True)
        
        # Show recent pitches with key metrics including Release Side
        display_cols = ['Date', 'Pitch Type', 'Velocity', 'Total Spin']
        if 'HB (trajectory)' in pitch_data.columns:
            display_cols.append('HB (trajectory)')
        if 'VB (trajectory)' in pitch_data.columns:
            display_cols.append('VB (trajectory)')
        if 'Release Height' in pitch_data.columns:
            display_cols.append('Release Height')
        if 'Release Side' in pitch_data.columns:
            display_cols.append('Release Side')
        if 'Is Strike' in pitch_data.columns:
            display_cols.append('Is Strike')
        
        recent_data = pitch_data[display_cols].head(20)
        
        st.dataframe(
            recent_data,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
                "Total Spin": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
                "HB (trajectory)": st.column_config.NumberColumn("H-Break", format="%.1f in"),
                "VB (trajectory)": st.column_config.NumberColumn("V-Break", format="%.1f in"),
                "Release Height": st.column_config.NumberColumn("Release Height", format="%.2f ft"),
                "Release Side": st.column_config.NumberColumn("Release Side", format="%.2f ft"),
                "Is Strike": st.column_config.TextColumn("Strike")
            }
        )

        display_player_force_plate_section(selected_player)

        # Add biomechanics section
        st.markdown('<h3 class="section-header">Biomechanical Analysis</h3>', unsafe_allow_html=True)
        biomechanics_display(selected_player)

        # Add player-specific assessment data
        display_player_assessment_data(selected_player)

    # Footer
    st.markdown("---")
    st.markdown("*St. Bonaventure Baseball Player Lookup | Made by Liquid Sports Lab*")

if __name__ == "__main__":
    main()