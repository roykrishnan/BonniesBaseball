import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from datetime import datetime, date, timedelta
import os
import glob
import requests 

# Set matplotlib style for dark mode
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Page configuration
st.set_page_config(
    page_title="St. Bonaventure Baseball Performance Dashboard", 
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
    .leaderboard-title {
        color: #54342c;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid #54342c;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #54342c;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(84, 52, 44, 0.1);
    }
    .rank-1 {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #54342c;
        font-weight: bold;
    }
    .rank-2 {
        background: linear-gradient(135deg, #C0C0C0 0%, #A9A9A9 100%);
        color: #54342c;
        font-weight: bold;
    }
    .rank-3 {
        background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%);
        color: white;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #54342c !important;
        border: 2px solid #54342c !important;
    }
    .stSelectbox > div > div > div {
        color: white !important;
    }
    .stSelectbox > div > div > div > div {
        color: white !important;
    }
    .stSelectbox span {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: white !important;
    }
            
    .stSelectbox label {
    color: white !important;
    font-weight: bold !important;}
            
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #54342c;
        border: 1px solid #54342c;
        border-radius: 4px 4px 0 0;
    }
    .stTab [data-baseweb="tab"][aria-selected="true"] {
        background-color: #54342c;
        color: #ffffff;
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
def load_rapsodo_data():
    """Load Rapsodo pitching data from CSV files in data directory"""
    data_dir = "data"
    all_player_data = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please ensure the data folder exists.")
    
    # Get all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}' directory.")
    
    for csv_file in csv_files:
        try:
            # Try different encodings to handle various file formats
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
                st.warning(f"Could not read file {csv_file} with any supported encoding")
                continue
            
            player_id = None
            player_name = None
            data_start_row = None
            
            # Find player info and data start
            for i, line in enumerate(lines):
                if 'Player ID:' in line:
                    player_id = line.split(',')[1].strip()
                elif 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                elif line.startswith('No,Date'):
                    data_start_row = i
                    break
            
            if data_start_row is not None and player_name and player_id:
                # Read the pitch data with the same encoding
                pitch_data = pd.read_csv(csv_file, skiprows=data_start_row, encoding=successful_encoding)
                
                # Filter out rows with missing pitch type
                pitch_data = pitch_data[pitch_data['Pitch Type'].notna()]
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '-']
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '']
                
                if len(pitch_data) > 0:
                    # Convert numeric columns
                    numeric_cols = ['Velocity', 'Total Spin', 'VB (trajectory)', 'HB (trajectory)', 'Release Height', 'Release Side', 'Horizontal Angle']
                    for col in numeric_cols:
                        if col in pitch_data.columns:
                            pitch_data[col] = pd.to_numeric(pitch_data[col], errors='coerce')
                    
                    # Calculate metrics for each pitch type
                    pitch_types = ['Fastball', 'ChangeUp', 'Slider']
                    player_stats = {
                        'PlayerID': player_id,
                        'PlayerName': player_name,
                        'TotalPitches': len(pitch_data)
                    }
                    
                    valid_pitch_types = []
                    
                    for pitch_type in pitch_types:
                        if pitch_type == 'Fastball':
                            pitch_data_filtered = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                        elif pitch_type == 'ChangeUp':
                            # Include both ChangeUp and Splitter in ChangeUp category
                            pitch_data_filtered = pitch_data[
                                pitch_data['Pitch Type'].str.contains('ChangeUp|Splitter', case=False, na=False)
                            ]
                        elif pitch_type == 'Slider':
                            # Include both Slider and Curveball in Slider category
                            pitch_data_filtered = pitch_data[
                                pitch_data['Pitch Type'].str.contains('Slider|Curveball', case=False, na=False)
                            ]
                        else:
                            pitch_data_filtered = pitch_data[pitch_data['Pitch Type'].str.contains(pitch_type, case=False, na=False)]
                        
                        if len(pitch_data_filtered) > 0:
                            valid_pitch_types.append(pitch_type)
                            
                            # Calculate metrics for this pitch type
                            avg_velocity = pitch_data_filtered['Velocity'].mean()
                            avg_spin_rate = pitch_data_filtered['Total Spin'].mean()
                            avg_release_height = pitch_data_filtered['Release Height'].mean() if 'Release Height' in pitch_data_filtered.columns else 5.5
                            avg_release_side = pitch_data_filtered['Release Side'].mean() if 'Release Side' in pitch_data_filtered.columns else 0
                            avg_vb = pitch_data_filtered['VB (trajectory)'].mean() if 'VB (trajectory)' in pitch_data_filtered.columns else 0
                            avg_hb = pitch_data_filtered['HB (trajectory)'].mean() if 'HB (trajectory)' in pitch_data_filtered.columns else 0
                            avg_horizontal_angle = pitch_data_filtered['Horizontal Angle'].mean() if 'Horizontal Angle' in pitch_data_filtered.columns else 0
                            
                            # Calculate speed difference vs their fastball
                            fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                            if len(fastball_data) > 0:
                                fastball_avg_velocity = fastball_data['Velocity'].mean()
                                speed_diff = fastball_avg_velocity - avg_velocity
                            else:
                                speed_diff = 0
                            
                            # Store metrics for this pitch type
                            player_stats.update({
                                f'{pitch_type}_Velocity': avg_velocity,
                                f'{pitch_type}_SpinRate': avg_spin_rate,
                                f'{pitch_type}_ReleaseHeight': avg_release_height,
                                f'{pitch_type}_ReleaseSide': avg_release_side,
                                f'{pitch_type}_HorizontalAngle': avg_horizontal_angle,
                                f'{pitch_type}_SpeedDiff': speed_diff,
                                f'{pitch_type}_HorizontalBreak': abs(avg_hb) if not pd.isna(avg_hb) else 0,
                                f'{pitch_type}_VerticalBreak': avg_vb if not pd.isna(avg_vb) else 0,
                                f'{pitch_type}_Pitches': len(pitch_data_filtered)
                            })
                    
                    # Only add player if they have at least one valid pitch type
                    if valid_pitch_types:
                        all_player_data.append(player_stats)
                    else:
                        st.warning(f"No valid pitch types found for {player_name} in {csv_file}")
                else:
                    st.warning(f"No valid pitch data found for {player_name} in {csv_file}")
            else:
                st.warning(f"Could not find player info or data start in {csv_file}")
        
        except Exception as e:
            st.error(f"Error reading file {csv_file}: {str(e)}")
            continue
    
    if not all_player_data:
        raise ValueError("No valid player data found in CSV files.")
    
    df = pd.DataFrame(all_player_data)
    
    # Calculate Bonnies Stuff+ for each pitch type
    pitch_types = ['Fastball', 'ChangeUp', 'Slider']
    
    for pitch_type in pitch_types:
        # Check if this pitch type has data
        velocity_col = f'{pitch_type}_Velocity'
        if velocity_col in df.columns:
            # Create temporary dataframe for this pitch type
            pitch_df = df[df[velocity_col].notna()].copy()
            if len(pitch_df) > 0:
                # Calculate Bonnies Stuff+ for this pitch type
                stuff_plus_col = f'{pitch_type}_Stuff+'
                pitch_df[stuff_plus_col] = calculate_bonnies_stuff_plus_for_pitch_type(pitch_df, pitch_type)
                
                # Merge back to main dataframe
                df = df.merge(pitch_df[['PlayerName', stuff_plus_col]], on='PlayerName', how='left')
    
    # Calculate Total Stuff+ as mean of all pitch types
    stuff_plus_cols = [col for col in df.columns if col.endswith('_Stuff+')]
    if stuff_plus_cols:
        df['Total_Stuff+'] = df[stuff_plus_cols].mean(axis=1, skipna=True)
    
    return df

def calculate_bonnies_stuff_plus_for_pitch_type(df, pitch_type):
    """Calculate Bonnies Stuff+ for a specific pitch type"""
    
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
    
    # Updated weights based on your specifications
    weights = {
        'velocity': 0.225,         # 22.5%
        'spin_rate': 0.175,        # 17.5%  
        'release_height': 0.125,   # 12.5%
        'release_side': 0.085,     # 8.5%
        'horizontal_angle': 0.05,  # 5%
        'speed_diff': 0.075,       # Reduced from 0.10 to 0.075 (7.5%)
        'horizontal_break': 0.10,  # 10%
        'vertical_break': 0.10,    # 10%
        'distinctive_shape': 0.125 # Increased from 0.10 to 0.125 (12.5%)
    }
    
    # Get column names for this pitch type
    velocity_col = f'{pitch_type}_Velocity'
    spin_col = f'{pitch_type}_SpinRate'
    height_col = f'{pitch_type}_ReleaseHeight'
    side_col = f'{pitch_type}_ReleaseSide'
    angle_col = f'{pitch_type}_HorizontalAngle'
    speed_diff_col = f'{pitch_type}_SpeedDiff'
    h_break_col = f'{pitch_type}_HorizontalBreak'
    v_break_col = f'{pitch_type}_VerticalBreak'
    
    # Calculate normalized components
    velocity_norm = normalize_component(df[velocity_col], higher_is_better=True)
    spin_norm = normalize_component(df[spin_col], higher_is_better=True)
    speed_diff_norm = normalize_component(df[speed_diff_col], higher_is_better=True)
    horizontal_angle_norm = normalize_component(abs(df[angle_col]), higher_is_better=False)
    h_break_norm = normalize_component(df[h_break_col], higher_is_better=True)
    v_break_norm = normalize_component(df[v_break_col], higher_is_better=True)
    
    # New components that reward deviation from mean
    height_norm = normalize_deviation_from_mean(df[height_col])
    
    # Handle release side if it exists, otherwise use zeros
    if side_col in df.columns:
        side_norm = normalize_deviation_from_mean(df[side_col])
    else:
        side_norm = np.ones(len(df)) * 0.5  # Neutral score if no data
    
    # Distinctive shape: reward pitches where |horizontal_break| - |vertical_break| deviates from 0
    shape_differential = np.abs(df[h_break_col]) - np.abs(df[v_break_col])
    distinctive_shape_norm = normalize_deviation_from_mean(np.abs(shape_differential))
    
    # Calculate weighted composite score
    composite_score = (
        velocity_norm * weights['velocity'] +
        spin_norm * weights['spin_rate'] +
        height_norm * weights['release_height'] +
        side_norm * weights['release_side'] +
        horizontal_angle_norm * weights['horizontal_angle'] +
        speed_diff_norm * weights['speed_diff'] +
        h_break_norm * weights['horizontal_break'] +
        v_break_norm * weights['vertical_break'] +
        distinctive_shape_norm * weights['distinctive_shape']
    )
    
    # Convert to industry-standard scale
    median_score = np.median(composite_score)
    std_score = np.std(composite_score)
    
    # Scale to Stuff+ where median = 100, std = 20
    stuff_plus = 100 + ((composite_score - median_score) / std_score) * 20
    
    # Cap at reasonable bounds
    stuff_plus = np.clip(stuff_plus, 40, 160)
    
    return stuff_plus

def create_leaderboard_chart(df, metric_col, title):
    """Create a horizontal bar chart for leaderboards using matplotlib dark mode"""
    df_sorted = df.sort_values(metric_col, ascending=False).head(10)  # Get top 10 performers
    df_sorted = df_sorted.sort_values(metric_col, ascending=True)  # Then sort ascending for horizontal bar display
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # St. Bonaventure color scheme
    primary_color = '#54342c'
    accent_colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887']
    
    # Create gradient-like effect with different shades
    colors = []
    values = df_sorted[metric_col].values
    max_val = values.max()
    min_val = values.min()
    
    for val in values:
        # Normalize value to [0, 1]
        normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        # Create color based on performance
        if normalized > 0.8:
            colors.append('#FFD700')  # Gold for top performers
        elif normalized > 0.6:
            colors.append('#CD853F')  # Tan
        elif normalized > 0.4:
            colors.append('#A0522D')  # Sienna
        else:
            colors.append('#8B4513')  # Saddle Brown
    
    # Create horizontal bar chart
    bars = ax.barh(df_sorted['PlayerName'], df_sorted[metric_col], color=colors, 
                   edgecolor='white', linewidth=1, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted[metric_col])):
        ax.text(bar.get_width() + max(df_sorted[metric_col]) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{value:.1f}', 
                va='center', ha='left', color='white', fontweight='bold', fontsize=10)
    
    # Styling
    ax.set_xlabel(metric_col.replace('_', ' '), color='white', fontsize=12, fontweight='bold')
    ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig

def create_comparison_boxplot(df, metric_col, group_col, title):
    """Create a boxplot comparison using matplotlib dark mode"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Get unique groups
    groups = df[group_col].unique()
    data_groups = [df[df[group_col] == group][metric_col].dropna() for group in groups]
    
    # Create boxplot
    bp = ax.boxplot(data_groups, labels=groups, patch_artist=True,
                    boxprops=dict(facecolor='#54342c', alpha=0.7),
                    medianprops=dict(color='#FFD700', linewidth=2),
                    whiskerprops=dict(color='white'),
                    capprops=dict(color='white'),
                    flierprops=dict(marker='o', markerfacecolor='#CD853F', markersize=6, alpha=0.7))
    
    # Color the boxes
    colors = ['#54342c', '#8B4513']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Styling
    ax.set_ylabel(metric_col.replace('_', ' '), color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel(group_col, color='white', fontsize=12, fontweight='bold')
    ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig

def create_leaderboard_table(df, metric_col, additional_cols=None):
    """Create a formatted leaderboard table"""
    df_sorted = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # Select columns for display
    display_cols = ['Rank', 'PlayerName', metric_col]
    if additional_cols:
        # Only add columns that actually exist in the dataframe
        for col in additional_cols:
            if col in df_sorted.columns:
                display_cols.append(col)
    
    return df_sorted[display_cols]

@st.cache_data
def load_individual_pitch_data():
    """Load individual pitch data with simple grading"""
    data_dir = "data"
    all_pitch_data = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
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
            # Read file with encoding handling
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
                    # Convert numeric columns
                    numeric_cols = ['Velocity', 'Total Spin', 'VB (trajectory)', 'HB (trajectory)', 'Release Height', 'Horizontal Angle']
                    for col in numeric_cols:
                        if col in pitch_data.columns:
                            pitch_data[col] = pd.to_numeric(pitch_data[col], errors='coerce')
                    
                    # Add player info and handedness
                    pitch_data['PlayerName'] = player_name
                    pitch_data['PitcherHand'] = handedness_map.get(player_name, 'RHP')
                    
                    # Calculate simple individual pitch grades
                    for idx, row in pitch_data.iterrows():
                        if pd.notna(row['Velocity']) and pd.notna(row['Total Spin']):
                            
                            # Simple grading based on velocity and spin
                            velocity_grade = min(100, max(20, row['Velocity'] * 1.2))  # Simple velocity scaling
                            spin_grade = min(100, max(20, row['Total Spin'] / 30))     # Simple spin scaling
                            
                            # Movement component
                            h_break = row['HB (trajectory)'] if pd.notna(row['HB (trajectory)']) else 0
                            v_break = row['VB (trajectory)'] if pd.notna(row['VB (trajectory)']) else 0
                            movement_grade = min(100, max(20, (abs(h_break) + abs(v_break)) * 5))
                            
                            # Simple composite grade
                            base_grade = (velocity_grade * 0.4 + spin_grade * 0.3 + movement_grade * 0.3)
                            
                            # Store individual pitch data
                            pitch_record = {
                                'PlayerName': player_name,
                                'PitcherHand': handedness_map.get(player_name, 'RHP'),
                                'PitchType': row['Pitch Type'],
                                'Velocity': row['Velocity'],
                                'TotalSpin': row['Total Spin'],
                                'HBreak': h_break,
                                'VBreak': v_break,
                                'VsRightyGrade': base_grade + (5 if handedness_map.get(player_name, 'RHP') == 'LHP' else 0),
                                'VsLeftyGrade': base_grade + (5 if handedness_map.get(player_name, 'RHP') == 'RHP' else 0),
                                'Date': row['Date'] if 'Date' in row else 'Unknown'
                            }
                            
                            all_pitch_data.append(pitch_record)
        
        except Exception as e:
            continue
    
    return pd.DataFrame(all_pitch_data) if all_pitch_data else pd.DataFrame()

# Header
st.markdown('<h1 class="main-header">St. Bonaventure Baseball</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Bonnies Stuff+ Dashboard</p>', unsafe_allow_html=True)

# Load data
try:
    rapsodo_df = load_rapsodo_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Check if we have data
if rapsodo_df.empty:
    st.error("No Rapsodo data loaded. Please check your data directory and CSV files.")
    st.info("Expected data directory: './data/' with CSV files containing Rapsodo pitch data.")
    st.stop()

# Sidebar with logo
try:
    st.sidebar.image("images/liquid_logo.png", width=250)
    st.sidebar.image("images/logo.png", width=250)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at images/logo.png")

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

# Pitch type selector
st.sidebar.subheader("Pitch Type Selection")
pitch_type_options = ["Total", "Fastball", "ChangeUp", "Slider"]
selected_pitch_type = st.sidebar.selectbox("Select Pitch Type", pitch_type_options)

# Determine the column name for selected pitch type
if selected_pitch_type == "Total":
    stuff_plus_col = "Total_Stuff+"
    display_name = "Total Stuff+"
else:
    stuff_plus_col = f"{selected_pitch_type}_Stuff+"
    display_name = f"{selected_pitch_type} Stuff+"

st.sidebar.markdown("---")

# Team overview metrics in sidebar
st.sidebar.subheader("Team Overview")
total_players = len(rapsodo_df)

# Show metrics for selected pitch type
if stuff_plus_col in rapsodo_df.columns:
    avg_stuff_plus = rapsodo_df[stuff_plus_col].mean()
    median_stuff_plus = rapsodo_df[stuff_plus_col].median()
    
    st.sidebar.metric("Total Players", total_players)
    st.sidebar.metric(f"Avg {display_name}", f"{avg_stuff_plus:.1f}")
    st.sidebar.metric(f"Median {display_name}", f"{median_stuff_plus:.1f}")
else:
    st.sidebar.metric("Total Players", total_players)
    st.sidebar.warning(f"No data for {display_name}")

total_pitches = rapsodo_df['TotalPitches'].sum()
st.sidebar.metric("Total Pitches", f"{total_pitches:,}")

# Main content
st.subheader(f"Bonnies {display_name} Leaderboard")

# Check if data exists for selected pitch type
if stuff_plus_col not in rapsodo_df.columns:
    st.error(f"No data available for {display_name}. Please select a different pitch type.")
    st.stop()

# Filter out players without data for this pitch type
display_df = rapsodo_df[rapsodo_df[stuff_plus_col].notna()].copy()

if len(display_df) == 0:
    st.error(f"No players have data for {display_name}")
    st.stop()

col1, col2 = st.columns([3, 2])

with col1:
    fig_stuff = create_leaderboard_chart(
        display_df, stuff_plus_col, 
        f"Bonnies {display_name} Rankings"
    )
    st.pyplot(fig_stuff, use_container_width=True)

with col2:
    st.subheader(f"Top Performers - {selected_pitch_type}")
    
    # Get relevant columns for this pitch type
    if selected_pitch_type == "Total":
        additional_cols = ['TotalPitches']
    else:
        additional_cols = [
            f'{selected_pitch_type}_Velocity',
            f'{selected_pitch_type}_SpinRate',
            f'{selected_pitch_type}_Pitches'
        ]
    
    stuff_table = create_leaderboard_table(
        display_df, stuff_plus_col, additional_cols
    )
    
    # Configure columns dynamically
    column_config = {
        stuff_plus_col: st.column_config.NumberColumn(display_name, format="%.1f")
    }
    
    if selected_pitch_type != "Total":
        column_config.update({
            f"{selected_pitch_type}_Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
            f"{selected_pitch_type}_SpinRate": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
            f"{selected_pitch_type}_Pitches": st.column_config.NumberColumn("Pitches", format="%.0f")
        })
    else:
        column_config["TotalPitches"] = st.column_config.NumberColumn("Total Pitches", format="%.0f")
    
    st.dataframe(
        stuff_table.head(10), 
        hide_index=True, 
        use_container_width=True,
        column_config=column_config
    )

# Full leaderboard
if selected_pitch_type == "Total":
    # Show all pitch types for Total view
    full_table_cols = ['TotalPitches']
    # Add available Stuff+ columns
    for pt in ['Fastball', 'ChangeUp', 'Slider']:
        if f'{pt}_Stuff+' in display_df.columns:
            full_table_cols.append(f'{pt}_Stuff+')
else:
    # Show detailed metrics for specific pitch type
    full_table_cols = [
        f'{selected_pitch_type}_Velocity',
        f'{selected_pitch_type}_SpinRate',
        f'{selected_pitch_type}_ReleaseHeight',
        f'{selected_pitch_type}_ReleaseSide',
        f'{selected_pitch_type}_HorizontalAngle',
        f'{selected_pitch_type}_SpeedDiff',
        f'{selected_pitch_type}_HorizontalBreak',
        f'{selected_pitch_type}_VerticalBreak',
        f'{selected_pitch_type}_Pitches'
    ]

full_table = create_leaderboard_table(display_df, stuff_plus_col, full_table_cols)

# Configure columns for full table
full_column_config = {
    stuff_plus_col: st.column_config.NumberColumn(display_name, format="%.1f")
}

if selected_pitch_type == "Total":
    full_column_config["TotalPitches"] = st.column_config.NumberColumn("Total Pitches", format="%.0f")
    for pt in ['Fastball', 'ChangeUp', 'Slider']:
        if f'{pt}_Stuff+' in display_df.columns:
            full_column_config[f'{pt}_Stuff+'] = st.column_config.NumberColumn(f"{pt} Stuff+", format="%.1f")
else:
    full_column_config.update({
        f"{selected_pitch_type}_Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
        f"{selected_pitch_type}_SpinRate": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
        f"{selected_pitch_type}_ReleaseHeight": st.column_config.NumberColumn("Release Height", format="%.2f ft"),
        f"{selected_pitch_type}_ReleaseSide": st.column_config.NumberColumn("Release Side", format="%.2f ft"),
        f"{selected_pitch_type}_HorizontalAngle": st.column_config.NumberColumn("H-Angle", format="%.1f°"),
        f"{selected_pitch_type}_SpeedDiff": st.column_config.NumberColumn("Speed Diff", format="%.1f mph"),
        f"{selected_pitch_type}_HorizontalBreak": st.column_config.NumberColumn("H-Break", format="%.1f in"),
        f"{selected_pitch_type}_VerticalBreak": st.column_config.NumberColumn("V-Break", format="%.1f in"),
        f"{selected_pitch_type}_Pitches": st.column_config.NumberColumn("Pitches", format="%.0f")
    })

st.dataframe(
    full_table, 
    hide_index=True, 
    use_container_width=True,
    column_config=full_column_config
)

# Right vs Left Handed Analysis
st.subheader(f"{display_name} - Right vs Left Handed Pitchers")

# Handedness mapping for St. Bonaventure Baseball team
handedness_map = {
    'Andrew Ayers': 'LHP',
    'Tom Wilkie': 'LHP', 
    'Conor Wolf': 'LHP',
    'Jakson Ross': 'LHP',
    'Mark Holm': 'LHP',
    'Nolan Feidt': 'LHP',
    'Ty Corey': 'LHP'
    # Everyone else defaults to RHP
}

# Add handedness to display_df - default to RHP for unlisted players
display_df['Handedness'] = display_df['PlayerName'].map(handedness_map).fillna('RHP')

col1, col2 = st.columns(2)

with col1:
    # Right-handed pitchers
    rhp_df = display_df[display_df['Handedness'] == 'RHP']
    if len(rhp_df) > 0:
        st.subheader(f"Right-Handed Pitchers ({len(rhp_df)})")
        
        # Create RHP leaderboard
        rhp_table = create_leaderboard_table(rhp_df, stuff_plus_col, [])
        st.dataframe(
            rhp_table.head(10),
            hide_index=True,
            use_container_width=True,
            column_config={
                stuff_plus_col: st.column_config.NumberColumn(display_name, format="%.1f")
            }
        )
        
        # RHP stats
        rhp_avg = rhp_df[stuff_plus_col].mean()
        rhp_median = rhp_df[stuff_plus_col].median()
        st.metric("RHP Average", f"{rhp_avg:.1f}")
        st.metric("RHP Median", f"{rhp_median:.1f}")
    else:
        st.info("No right-handed pitchers with data for this pitch type")

with col2:
    # Left-handed pitchers
    lhp_df = display_df[display_df['Handedness'] == 'LHP']
    if len(lhp_df) > 0:
        st.subheader(f"Left-Handed Pitchers ({len(lhp_df)})")
        
        # Create LHP leaderboard
        lhp_table = create_leaderboard_table(lhp_df, stuff_plus_col, [])
        st.dataframe(
            lhp_table.head(10),
            hide_index=True,
            use_container_width=True,
            column_config={
                stuff_plus_col: st.column_config.NumberColumn(display_name, format="%.1f")
            }
        )
        
        # LHP stats
        lhp_avg = lhp_df[stuff_plus_col].mean()
        lhp_median = lhp_df[stuff_plus_col].median()
        st.metric("LHP Average", f"{lhp_avg:.1f}")
        st.metric("LHP Median", f"{lhp_median:.1f}")
    else:
        st.info("No left-handed pitchers with data for this pitch type")

# Set matplotlib to dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

# VALD API Configuration
VALD_CONFIG = st.secrets["VALD_CONFIG"]

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
def load_bonnies_players_from_csv():
    """Load St. Bonaventure Baseball players from CSV files"""
    data_dir = "data"
    bonnies_players = {}
    
    if not os.path.exists(data_dir):
        return {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    handedness_map = {
        'Andrew Ayers': 'LHP', 'Tom Wilkie': 'LHP', 'Conor Wolf': 'LHP',
        'Jakson Ross': 'LHP', 'Mark Holm': 'LHP', 'Nolan Feidt': 'LHP', 'Ty Corey': 'LHP'
    }
    
    for csv_file in csv_files:
        try:
            encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
            lines = None
            
            for encoding in encodings_to_try:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                continue
            
            player_id = None
            player_name = None
            
            for line in lines:
                if 'Player ID:' in line:
                    player_id = line.split(',')[1].strip()
                elif 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                    break
            
            if player_id and player_name:
                handedness = handedness_map.get(player_name, 'RHP')
                bonnies_players[player_name] = {
                    'player_id': player_id,
                    'handedness': handedness,
                    'csv_file': csv_file
                }
                
        except Exception:
            continue
    
    return bonnies_players

@st.cache_data(ttl=1800)
def fetch_all_profiles():
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

def match_players_to_profiles(bonnies_players, all_profiles):
    """Match CSV players to VALD profiles by name"""
    name_to_profile_id = {}
    
    for player_name in bonnies_players.keys():
        for profile_id, profile_data in all_profiles.items():
            if profile_data['fullName'] == player_name:
                name_to_profile_id[player_name] = profile_id
                break
    
    return name_to_profile_id

@st.cache_data(ttl=1800)
def get_team_id():
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
def fetch_forcedecks_tests(profile_ids, modified_from_date):
    """Fetch ForceDecks test data using /tests endpoint"""
    if not profile_ids:
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
                        filtered_tests = [test for test in tests if test.get('profileId') in profile_ids]
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
                st.error(f"API Error {response.status_code}")
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
def fetch_test_trials_batch(team_id, test_ids):
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

def extract_performance_metrics_from_trials(trials_df, test_data):
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

def create_leaderboard_dashboard(perf_df, bonnies_players):
    """Create focused leaderboard dashboard for the four key tests"""
    
    if perf_df.empty:
        st.warning("No performance data available")
        return
    
    # Add player names and handedness
    profile_id_to_name = {}
    for name, info in bonnies_players.items():
        # Match by athleteId or profileId
        for _, row in perf_df.iterrows():
            athlete_id = row.get('athleteId') or row.get('profileId')
            if athlete_id and athlete_id in [info.get('player_id'), name]:
                profile_id_to_name[athlete_id] = name
                break
    
    # If direct matching didn't work, try profile matching
    if not profile_id_to_name:
        # This would need the profile matching from session state
        if 'name_to_profile_id' in st.session_state:
            profile_id_to_name = {v: k for k, v in st.session_state.name_to_profile_id.items()}
    
    perf_df['player_name'] = perf_df['profileId'].map(profile_id_to_name)
    
    # Add handedness
    name_to_handedness = {name: info['handedness'] for name, info in bonnies_players.items()}
    perf_df['handedness'] = perf_df['player_name'].map(name_to_handedness)
    
    # Clean up test types - fix SLJ to SJ
    perf_df['testType'] = perf_df['testType'].replace('SLJ', 'SJ')
    
    # Map actual test codes to display names
    test_mapping = {
        'CMJ': 'CMJ',
        'SJ': 'Squat Jump', 
        'HJ': 'Hop Jump',
        'PPU': 'Plyo Pushup'
    }
    
    # Focus on the four key tests using actual codes
    target_test_codes = ['CMJ', 'SJ', 'HJ', 'PPU']
    available_tests = perf_df['testType'].unique()
    
    st.subheader("Force Plate Test Leaderboards")
    
    # Filter for target tests that are available
    filtered_test_codes = [test for test in target_test_codes if test in available_tests]
    
    if not filtered_test_codes:
        st.warning(f"None of the target tests ({target_test_codes}) found in data. Available: {list(available_tests)}")
        # Show all available tests for debugging
        st.write("Creating tabs for all available test types:")
        filtered_test_codes = list(available_tests)
        test_mapping.update({code: code for code in available_tests if code not in test_mapping})
    
    # Create tabs for each test type using display names
    tab_names = [test_mapping.get(code, code) for code in filtered_test_codes]
    tabs = st.tabs(tab_names)
    
    for i, test_code in enumerate(filtered_test_codes):
        with tabs[i]:
            display_name = test_mapping.get(test_code, test_code)
            create_test_leaderboard(perf_df, test_code, display_name)

def create_test_leaderboard(perf_df, test_code, display_name):
    """Create leaderboard for a specific test type"""
    
    test_data = perf_df[perf_df['testType'] == test_code].copy()
    
    if test_data.empty:
        st.warning(f"No data found for {display_name} ({test_code})")
        return
    
    st.subheader(f"{display_name} Performance Analysis")
    
    # Key metrics for each test type (using codes)
    key_metrics = {
        'CMJ': ['Jump Height (Flight Time)', 'Peak Power','Takeoff Peak Force', 'RSI-modified'],
        'SJ': ['Jump Height (Flight Time)', 'Peak Power', 'Takeoff Peak Force'],
        'PPU': ['Peak Power', 'Peak Force', 'Flight Time'],
        'HJ': ['Jump Height (Flight Time)', 'Peak Force', 'Landing RFD']
    }
    
    available_metrics = test_data['metric_name'].unique()
    target_metrics = key_metrics.get(test_code, [])
    
    # Find metrics that match our targets (partial matching)
    matched_metrics = []
    for target in target_metrics:
        for available in available_metrics:
            if target.lower() in available.lower() or available.lower() in target.lower():
                matched_metrics.append(available)
                break
    
    if not matched_metrics:
        st.warning(f"No key metrics found for {display_name}")
        st.write(f"Available metrics: {list(available_metrics)[:10]}...")  # Show first 10
        return
    
    # Create metric selector
    selected_metric = st.selectbox(f"Select {display_name} Metric:", matched_metrics)
    
    # Filter for the selected metric and trial limb only (best overall performance)
    metric_data = test_data[
        (test_data['metric_name'] == selected_metric) & 
        (test_data['limb'] == 'Trial')  # Overall trial result, not left/right/asymmetry
    ].copy()
    
    if metric_data.empty:
        st.warning(f"No data found for {selected_metric}")
        return
    
    # Automatic outlier removal using IQR method (3x multiplier for obvious outliers)
    Q1 = metric_data['value'].quantile(0.25)
    Q3 = metric_data['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Count outliers before removal
    outliers = metric_data[(metric_data['value'] < lower_bound) | (metric_data['value'] > upper_bound)]
    outlier_count = len(outliers)
    
    # Remove outliers
    filtered_metric_data = metric_data[
        (metric_data['value'] >= lower_bound) & 
        (metric_data['value'] <= upper_bound)
    ].copy()
    
    # Show info if outliers were removed
    if outlier_count > 0:
        st.info(f"Automatically removed {outlier_count} obvious outlier(s) for cleaner analysis")
    
    # Get the best performance per player from cleaned data
    player_best = filtered_metric_data.groupby('player_name')['value'].max().reset_index()
    player_best = player_best.sort_values('value', ascending=False)
    
    # Add handedness and additional info
    if 'handedness' in filtered_metric_data.columns:
        handedness_map = filtered_metric_data.groupby('player_name')['handedness'].first().to_dict()
        player_best['handedness'] = player_best['player_name'].map(handedness_map)
    
    # Get metric info
    sample_metric = filtered_metric_data.iloc[0]
    units = sample_metric.get('units', '')
    description = sample_metric.get('description', '')
    
    # Display metric info
    st.info(f"**{selected_metric}** ({units}): {description}")
    
    # Statistics
    group_avg = player_best['value'].mean()
    group_std = player_best['value'].std()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Group Average", f"{group_avg:.2f} {units}")
    
    with col2:
        above_avg = len(player_best[player_best['value'] >= group_avg])
        st.metric("Above Average", f"{above_avg}/{len(player_best)}")
    
    with col3:
        if len(player_best) > 0:
            best_performer = player_best.iloc[0]['player_name']
            st.metric("Top Performer", best_performer.split()[0])
        else:
            st.metric("Top Performer", "N/A")
    
    with col4:
        cv = (group_std / group_avg) * 100 if group_avg != 0 else 0
        st.metric("Coefficient of Variation", f"{cv:.1f}%")
    
    # Leaderboard visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color coding based on performance
    colors = []
    for _, row in player_best.iterrows():
        value = row['value']
        if value >= group_avg + 0.5 * group_std:
            colors.append('#2E8B8B')  # Teal for excellent
        elif value >= group_avg:
            colors.append('#4A90A4')  # Blue for above average
        elif value >= group_avg - 0.5 * group_std:
            colors.append('#FFA500')  # Orange for below average
        else:
            colors.append('#FF6B6B')  # Red for needs improvement
    
    bars = ax.bar(range(len(player_best)), player_best['value'], color=colors, alpha=0.8)
    
    # Add group average line
    ax.axhline(y=group_avg, color='white', linestyle='--', linewidth=2, alpha=0.8, 
              label=f'Group Average: {group_avg:.2f} {units}')
    
    # Styling
    ax.set_title(f'St. Bonaventure Baseball\n{display_name} - {selected_metric}', 
                fontsize=16, pad=20, fontweight='bold')
    ax.set_ylabel(f'{selected_metric} ({units})', fontsize=12)
    ax.set_xlabel('Players (Ranked by Performance)', fontsize=12)
    
    # X-axis labels (first names only)
    ax.set_xticks(range(len(player_best)))
    ax.set_xticklabels([name.split()[0] for name in player_best['player_name']], 
                      rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, player_best['value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Detailed rankings table
    st.subheader(f"{display_name} - {selected_metric} Rankings")
    
    rankings_df = player_best.copy()
    rankings_df['rank'] = range(1, len(rankings_df) + 1)
    rankings_df['vs_average'] = ((rankings_df['value'] - group_avg) / group_avg * 100).round(1)
    rankings_df['percentile'] = [100 - (i/len(rankings_df))*100 for i in range(len(rankings_df))]
    
    # Display table
    display_columns = ['rank', 'player_name', 'value', 'vs_average', 'percentile']
    column_names = ['Rank', 'Player', f'Best ({units})', 'vs Avg (%)', 'Percentile']
    
    if 'handedness' in rankings_df.columns:
        display_columns.append('handedness')
        column_names.append('Hand')
    
    display_df = rankings_df[display_columns].copy()
    display_df.columns = column_names
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Test-specific training interpretation guide
    st.markdown("---")
    st.markdown("## Training Interpretation Guide")
    
    if test_code == 'CMJ':
        st.markdown("""
        ### CMJ (Countermovement Jump) Training Recommendations
        **Jump Height (Flight Time):**
        - **Below Average:** Focus on explosive power development with **explosive plyometric exercises** (box jumps, depth jumps, tuck jumps)
        
        **Peak Power:** 
        - **Low values:** Need **power-speed training** (medicine ball throws, jump squats, lift derivatives)
        
        **Peak Force:** 
        - **Below average:** **Strength deficits** - emphasize **heavy compound movements** (squats, deadlifts, hip thrusts)
        
        **RSI-modified:** 
        - **Poor reactive strength:** Requires **reactive plyometrics** (pogos, quick ground contacts, hurdle hops)
        """)
    
    elif test_code == 'SJ':
        st.markdown("""
        ### Squat Jump (SJ) Training Recommendations
        **Jump Height:** 
        - **Below Average:** **Concentric power deficits** - focus on **pause squats, quarter squats with heavy load**
        
        **Peak Power:** 
        - **Low values:** Need **ballistic training** (jump squats, medicine ball slams, explosive bench press)
        
        **Peak Force:** 
        - **Poor scores:** Require **maximal strength training** (1-5 rep range compound movements)
        
        **Training Parameters:**
        - **Strength focus:** Reps at 85-95% 1RM
        - **Power focus:** Reps at 30-60% 1RM with pause and explosive intent
        - **Concentric emphasis:** Remove eccentric component with pause squats and pin squats
        """)
    
    elif test_code == 'HJ':
        st.markdown("""
        ### Hop Jump (HJ) Training Recommendations
        **Jump Height:**
        - **Below Average:** Single-leg power deficits - emphasize **unilateral plyometrics** (single-leg bounds, lateral hops)
        
        **Peak Force:** 
        - **Low values:** **Single-leg strength needs** (Bulgarian split squats, single-leg RDLs, step-ups)
        
        **Landing RFD:** 
        - **Poor landing mechanics:** Require **eccentric control training** (landing drills, eccentric squats)
        
        **Training Parameters:**
        - **Unilateral strength:** 3-4 sets × 6-10 reps per leg with challenging load
        - **Single-leg power:** 3-5 sets × 3-6 reps per leg with explosive intent
        - **Landing mechanics:** Focus on controlled landings with 2-3 second holds
        """)
    
    elif test_code == 'PPU':
        st.markdown("""
        ### Plyo Pushup (PPU) Training Recommendations
        **Peak Power:**
        - **Below Average:** Upper body explosive deficit - focus on **upper body plyometrics** (plyo pushups, medicine ball chest passes, clap pushups)
        
        **Peak Force:** 
        - **Low values:** **Upper body strength needs** (bench press, weighted pushups, dips)
        
        **Flight Time:** 
        - **Poor airborne time:** Requires **explosive pushing power** (ballistic bench press, speed pushups)
        
        **Training Parameters:**
        - **Upper body strength:** Reps at 80-90% 1RM
        - **Upper body power:** Reps at 30-50% 1RM with maximal speed
        - **Plyometric progression:** Start with incline variations, progress to decline
        """)
    
    st.markdown("""
    ---
    *Note: All training recommendations should be implemented under qualified supervision with proper progression and recovery protocols.*
    """)

def main():
    st.title("St. Bonaventure Baseball - Force Plate Leaderboards")
    st.markdown('<p class="sub-header">Performance rankings for CMJ, Squat Jump, Plyo Pushup, and Hop Test</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'bonnies_players' not in st.session_state:
        st.session_state.bonnies_players = {}
    if 'all_profiles' not in st.session_state:
        st.session_state.all_profiles = {}
    if 'name_to_profile_id' not in st.session_state:
        st.session_state.name_to_profile_id = {}
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    # Initialize data
    if not st.session_state.initialized:
        with st.spinner("Loading team data..."):
            bonnies_players = load_bonnies_players_from_csv()
            all_profiles = fetch_all_profiles()
            name_to_profile_id = match_players_to_profiles(bonnies_players, all_profiles)
            
            st.session_state.bonnies_players = bonnies_players
            st.session_state.all_profiles = all_profiles
            st.session_state.name_to_profile_id = name_to_profile_id
            st.session_state.initialized = True
    
    st.sidebar.subheader("Force Plate Team Leaderboards")
    with st.sidebar.expander("Team Info", expanded=True):
        st.write(f"**Players with Motion Capture Data:** {len(st.session_state.bonnies_players)}")
        st.write(f"**Matched with VALD:** {len(st.session_state.name_to_profile_id)}")
    
    # Date selection
    selected_date = st.date_input(
        "Select Testing Date (First Test 2025/09/06)",
        value=date(2025, 9, 6),
        min_value=date(2024, 1, 1),
        max_value=date.today()
    )
    
    if st.button("Load Force Plate Data", type="primary"):
        profile_ids = list(st.session_state.name_to_profile_id.values())
        team_id = get_team_id()
        
        with st.spinner("Loading force plate data..."):
            # Get test summaries
            df = fetch_forcedecks_tests(profile_ids, selected_date.strftime('%Y-%m-%d'))
            
            if not df.empty:
                # Filter for selected date
                df_filtered = df[df['date'] == selected_date].copy()
                
                if not df_filtered.empty:
                    # Get trial data
                    test_ids = df_filtered['testId'].unique().tolist()
                    trials_df = fetch_test_trials_batch(team_id, test_ids)
                    
                    if not trials_df.empty:
                        # Extract performance metrics
                        perf_df = extract_performance_metrics_from_trials(trials_df, df_filtered)
                        
                        if not perf_df.empty:
                            st.session_state.performance_data = perf_df
                            st.success(f"Loaded {len(perf_df)} performance measurements from {len(df_filtered)} tests")
                        else:
                            st.error("No performance metrics extracted from trial data")
                    else:
                        st.error("No trial data found")
                else:
                    st.warning(f"No test data found for {selected_date}")
            else:
                st.warning("No test data found")
    
    # Display leaderboards if data is available
    if 'performance_data' in st.session_state and not st.session_state.performance_data.empty:
        create_leaderboard_dashboard(st.session_state.performance_data, st.session_state.bonnies_players)
    else:
        st.info("Click 'Load Force Plate Data' to generate leaderboards for the selected date")

if __name__ == "__main__":
    main()

st.header("Table Assessments - Team View")
st.markdown('<p class="sub-header">Bonnies Baseball Assessment Table by Player</p>', unsafe_allow_html=True)
excel_file_path = os.path.join("data", "BonniesBaseballTableAssessment.xlsx")

try:
    # Read the Excel file
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    
    # Get columns A through V (first 22 columns)
    if df.shape[1] >= 22:
        display_df = df.iloc[:, 0:22].copy()  # Columns A through V
    else:
        display_df = df.copy()
        st.warning(f"File only contains {df.shape[1]} columns, displaying all available columns.")
    
    # Clean up the dataframe - remove any completely empty rows
    display_df = display_df.dropna(how='all')
    
    # Function to highlight outliers (values 2 std devs from median)
    def highlight_outliers(val, median, std, column_name):
        """
        Highlight values that are 2 standard deviations away from the median
        """
        # Skip non-numeric values
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''
        
        # Calculate bounds
        lower_bound = median - 2 * std
        upper_bound = median + 2 * std
        
        # Flag outliers
        if val < lower_bound:
            return 'background-color: #cc0000; color: white; font-weight: bold'  # Dark red for low outliers
        elif val > upper_bound:
            return 'background-color: #006400; color: white; font-weight: bold'  # Dark green for high outliers
        else:
            return ''
    
    # Calculate statistics for numeric columns
    stats_info = {}
    for column in display_df.columns:
        # Try to convert to numeric, skip if it fails
        numeric_col = pd.to_numeric(display_df[column], errors='coerce')
        
        # Only process if we have at least 3 numeric values
        if numeric_col.notna().sum() >= 3:
            median = numeric_col.median()
            std = numeric_col.std()
            
            if std > 0:  # Only flag if there's variation
                stats_info[column] = {
                    'median': median,
                    'std': std,
                    'lower_bound': median - 2 * std,
                    'upper_bound': median + 2 * std
                }
    
    # Apply styling
    if stats_info:
        def apply_outlier_styling(row):
            styles = [''] * len(row)
            for idx, (column, val) in enumerate(row.items()):
                if column in stats_info:
                    styles[idx] = highlight_outliers(
                        val, 
                        stats_info[column]['median'], 
                        stats_info[column]['std'],
                        column
                    )
            return styles
        
        styled_df = display_df.style.apply(apply_outlier_styling, axis=1)
        
        # Display legend with black background and darker colors
        st.markdown("""
        <div style='padding: 10px; background-color: #000000; border-radius: 5px; margin-bottom: 10px; color: white;'>
            <strong>Legend:</strong> 
            <span style='background-color: #cc0000; color: white; padding: 2px 8px; margin: 0 5px; border-radius: 3px; font-weight: bold;'>Mobility Deficiency (< Median - 2 SD)</span>
            <span style='background-color: #006400; color: white; padding: 2px 8px; margin: 0 5px; border-radius: 3px; font-weight: bold;'>Hyper Mobility (> Median + 2 SD)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the styled table
        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        # Show statistics summary
        with st.expander("View Column Statistics", expanded=False):
            stats_df = pd.DataFrame(stats_info).T
            stats_df = stats_df.round(2)
            st.dataframe(stats_df, use_container_width=True)
        
        # Training Translation Section
        st.markdown("---")
        st.markdown('<h3 class="section-header">Training Translation Guide</h3>', unsafe_allow_html=True)
        
        st.markdown("""
            Training translation: 
            If player shows restricted range of motion. Focus on mobility work, dynamic stretching, and tissue quality exercises to improve movement capacity. Prioritize addressing movement restrictions before adding load or intensity.</p>

                - Increase mobility drills and dynamic warm-ups
                - Address tissue quality (foam rolling, soft tissue work)
                - Focus on controlled articular rotations (CARs)
                - Gradually expand range of motion through progressive stretching
        """, unsafe_allow_html=True)
        
        st.markdown("""
        Training Translation:
         If player demonstrates excellent range of motion. Ensure adequate strength development to support and control this mobility, particularly at end ranges of motion.</p>

                - Emphasize strength training throughout full range of motion
                - Focus on eccentric control and end-range strength
                - Implement tempo work and isometric holds
                - Develop motor control to utilize available mobility effectively 
        """, unsafe_allow_html=True)
    else:
        # If no numeric columns found, display without styling
        st.warning("No numeric columns found for outlier detection. Displaying table without highlighting.")
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=600
        )

except Exception as e:
    st.error(f"Error reading Excel file: {str(e)}")
    st.info("Please ensure the file is a valid Excel file (.xlsx format)")

# Footer
st.markdown("---")
st.markdown("*St. Bonaventure Baseball Home Dashboard | Built by Liquid Sports Lab*")