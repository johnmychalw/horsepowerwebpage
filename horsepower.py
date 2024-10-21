
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import quantile_transform
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from io import BytesIO

# Load data
data = pd.read_csv('CleanHPdata2.1.csv')

# Ensure 'Age' is treated as numeric
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Metrics to compare
metrics = ['Grip Strength (Bottom Hand)', 'Grip Strength (Top Hand)', 'Vertical Jump', 'Med Ball SitUp', 'Med Ball Chest']
all_metrics = metrics + ['Horsepower']

# Ensure all metrics are numeric
for metric in metrics:
    data[metric] = pd.to_numeric(data[metric], errors='coerce').fillna(0)

# Define levels and positions
levels = ['High School', 'College', 'Minors', 'MLB']
positions = sorted(data['Position'].dropna().unique()) + ['Middle Infield', 'Corner Infield']

def get_percentiles_within_group(data, columns):
    percentiles = pd.DataFrame(index=data.index)
    for column in columns:
        numeric_data = pd.to_numeric(data[column], errors='coerce').fillna(0)
        percentiles[column] = quantile_transform(numeric_data.values.reshape(-1, 1), output_distribution='uniform').flatten()
    return percentiles

def plot_radar(input_data, comparison_data, categories, player_name, comparison_label, note=""):
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    input_data = np.concatenate((input_data, [input_data[0]]))
    comparison_data = np.concatenate((comparison_data, [comparison_data[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, input_data, color='lightcoral', alpha=0.25, label=f"{player_name}'s Data" if player_name else 'Input Data')
    ax.fill(angles, comparison_data, color='cornflowerblue', alpha=0.25, label=comparison_label)
    ax.plot(angles, input_data, color='lightcoral', linewidth=2, linestyle='solid')
    ax.plot(angles, comparison_data, color='cornflowerblue', linewidth=2, linestyle='solid')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='black')

    ax.yaxis.set_ticks(np.linspace(0, 1, 6))
    ax.yaxis.set_ticklabels(['0', '20', '40', '60', '80', '100'], color='grey')

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if note:
        plt.figtext(0.5, 0.02, note, ha="center", fontsize=8, color="grey")

    # Use BytesIO to pass the image to Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def plot_metric(metric, player_percentile, player_value):
    color = plt.cm.coolwarm(player_percentile / 100)
    fig, ax = plt.subplots(figsize=(6, 1))  # Adjusted width and height
    ax.barh(' ', player_percentile, color=color, edgecolor='black')
    ax.text(110, 0, f"{player_value:.2f}", va='center', ha='left', color='black', fontsize=8)  # Player value
    ax.text(player_percentile - 6, 0, f"{round(player_percentile)}", va='center', ha='left', color='white', fontsize=8)  # Rounded percentile value at the right end
    ax.set_xlim(0, 100)
    ax.set_title(metric, fontsize=10, color='black')
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

    # Use BytesIO to pass the image to Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# Input Player Data
st.header("Input Player Information")

# First row: Player Name
player_name = st.text_input("Player Name")

# Second row: Grip Strength inputs
col1, col2 = st.columns(2)
with col1:
    grip_strength_bottom = st.number_input("Grip Strength (Bottom Hand)", min_value=0.0)
with col2:
    grip_strength_top = st.number_input("Grip Strength (Top Hand)", min_value=0.0)

# Third row: Vertical Jump, Med Ball Situp, Med Ball Chest
col3, col4, col5 = st.columns(3)
with col3:
    vertical_jump = st.number_input("Vertical Jump", min_value=0.0)
with col4:
    med_ball_situp = st.number_input("Med Ball SitUp", min_value=0.0)
with col5:
    med_ball_chest = st.number_input("Med Ball Chest", min_value=0.0)

# Create a dictionary of input metrics
input_metrics = {
    'Grip Strength (Bottom Hand)': grip_strength_bottom,
    'Grip Strength (Top Hand)': grip_strength_top,
    'Vertical Jump': vertical_jump,
    'Med Ball SitUp': med_ball_situp,
    'Med Ball Chest': med_ball_chest
}

# Ensure that all inputs have been entered before proceeding
if all([player_name, grip_strength_bottom, grip_strength_top, vertical_jump, med_ball_situp, med_ball_chest]):
    
    # Calculate Horsepower for input data
    horsepower = vertical_jump + med_ball_situp + med_ball_chest
    input_data = list(input_metrics.values()) + [horsepower]

    # Create tabs for different comparison modes
    tabs = st.tabs(["Compare to Level", "Compare to Player", "Find Closest Match", "Compare to Position"])

    # 1. Compare to Level
    with tabs[0]:
        st.header("Compare to Level")

        # Select Group By options
        group_by = st.selectbox("Group By", ['Level', 'Age'])

        if group_by == 'Level':
            group_value = st.selectbox("Select Level ", levels)
        else:
            group_value = st.text_input(f"Select {group_by}")

        # Convert group_value to numeric if comparing by Age
        if group_by == 'Age':
            group_value = pd.to_numeric(group_value, errors='coerce')

        if group_value:  # Auto-generate graphs when user inputs the data
            comparison_group = data[data[group_by] == group_value]

            if not comparison_group.empty:
                group_percentiles = get_percentiles_within_group(comparison_group, all_metrics)

                input_data_percentiles = []
                for i, metric in enumerate(all_metrics):
                    metric_values = pd.to_numeric(comparison_group[metric], errors='coerce').fillna(0)
                    percentile_value = np.sum(metric_values < input_data[i]) / len(metric_values)
                    input_data_percentiles.append(percentile_value)

                comparison_data = group_percentiles.mean().values
                comparison_label = f"{group_value} {group_by} Average"

                radar_img = plot_radar(input_data_percentiles, comparison_data, all_metrics, player_name, comparison_label)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(radar_img)
                with col2:
                    for i, metric in enumerate(all_metrics):
                        bar_img = plot_metric(metric, input_data_percentiles[i] * 100, input_data[i])
                        st.image(bar_img, use_column_width=True)
            else:
                st.error("No data found for the specified group.")
        else:
             st.warning("Please fill out all the input fields to generate graphs.")

    with tabs[1]:
        st.header("Compare to Player")
    
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        
        # Ensure both player names and all inputs are provided before proceeding
        if first_name and last_name and all([grip_strength_bottom, grip_strength_top, vertical_jump, med_ball_situp, med_ball_chest]):
            compare_player = data[(data['First Name'] == first_name) & (data['Last Name'] == last_name)]
    
            if compare_player.empty:
                st.error("No player found with the specified name.")
            else:
                compare_player = compare_player.iloc[0]
                level = compare_player['Level']
                comparison_group = data[data['Level'] == level]
    
                if not comparison_group.empty:
                    group_percentiles = get_percentiles_within_group(comparison_group, all_metrics)
    
                    input_data_percentiles = []
                    for i, metric in enumerate(all_metrics):
                        metric_values = pd.to_numeric(comparison_group[metric], errors='coerce').fillna(0)
                        percentile_value = np.sum(metric_values < input_data[i]) / len(metric_values)
                        input_data_percentiles.append(percentile_value)
    
                    compare_player_percentiles = []
                    for i, metric in enumerate(all_metrics):
                        metric_values = pd.to_numeric(comparison_group[metric], errors='coerce').fillna(0)
                        percentile_value = np.sum(metric_values < compare_player[metric]) / len(metric_values)
                        compare_player_percentiles.append(percentile_value)
    
                    radar_img = plot_radar(input_data_percentiles, compare_player_percentiles, all_metrics, player_name, f"{first_name} {last_name}'s Data")
    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(radar_img)
                    with col2:
                        for i, metric in enumerate(all_metrics):
                            bar_img = plot_metric(metric, input_data_percentiles[i] * 100, input_data[i])
                            st.image(bar_img, use_column_width=True)
                else:
                    st.error("No data found for the specified level.")
        else:
            st.warning("Please fill in all required player inputs and the player name.")
    
    with tabs[2]:
        st.header("Find Closest Match")
        
        # Ensure that all inputs are provided before proceeding
        if all([grip_strength_bottom, grip_strength_top, vertical_jump, med_ball_situp, med_ball_chest]):
            # Collect input data
            input_data = [grip_strength_bottom, grip_strength_top, vertical_jump, med_ball_situp, med_ball_chest]
    
            # Calculate 'Horsepower'
            horsepower = vertical_jump + med_ball_situp + med_ball_chest
            input_data.append(horsepower)
    
            # Find the closest match in the dataset using Euclidean distance
            data_subset = data[all_metrics].dropna()
            distances = data_subset.apply(lambda row: euclidean(input_data, row), axis=1)
            closest_index = distances.idxmin()
            closest_match = data.loc[closest_index]
    
            # Get the level of the closest match
            level = closest_match['Level']
            comparison_group = data[data['Level'] == level]
    
            group_percentiles = get_percentiles_within_group(comparison_group, all_metrics)
    
            input_data_percentiles = []
            for i, metric in enumerate(all_metrics):
                metric_values = pd.to_numeric(comparison_group[metric], errors='coerce').fillna(0)
                percentile_value = np.sum(metric_values < input_data[i]) / len(metric_values)
                input_data_percentiles.append(percentile_value)
    
            closest_match_percentiles = []
            for i, metric in enumerate(all_metrics):
                metric_values = pd.to_numeric(comparison_group[metric], errors='coerce').fillna(0)
                percentile_value = np.sum(metric_values < closest_match[metric]) / len(metric_values)
                closest_match_percentiles.append(percentile_value)
    
            radar_img = plot_radar(input_data_percentiles, closest_match_percentiles, all_metrics, player_name, f"Closest Match: {closest_match['First Name']} {closest_match['Last Name']}")
    
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(radar_img)
            with col2:
                for i, metric in enumerate(all_metrics):
                    bar_img = plot_metric(metric, input_data_percentiles[i] * 100, input_data[i])
                    st.image(bar_img, use_column_width=True)
        else:
            st.warning("Please fill in all required player inputs.")

    with tabs[3]:
        st.header("Compare to Position")
        
        # Select Level and Position
        level = st.selectbox("Select Level", levels)
        position = st.selectbox("Select Position", positions)
        
        # Ensure that level, position, and all inputs are provided before proceeding
        if level and position and all([grip_strength_bottom, grip_strength_top, vertical_jump, med_ball_situp, med_ball_chest]):
            # Adjust positions for custom categories
            if position == 'Middle Infield':
                position_data = data[data['Position'].isin(['Shortstop', 'Second Base'])]
            elif position == 'Corner Infield':
                position_data = data[data['Position'].isin(['Third Base', 'First Base'])]
            else:
                position_data = data[data['Position'] == position]
    
            # Filter data by level
            level_data = data[data['Level'] == level]
            position_data = position_data[position_data['Level'] == level]
    
            if not position_data.empty:
                level_percentiles = get_percentiles_within_group(level_data, all_metrics)
                position_avg_percentiles = position_data[all_metrics].mean()
    
                position_percentiles = []
                for i, metric in enumerate(all_metrics):
                    metric_values = pd.to_numeric(level_data[metric], errors='coerce').fillna(0)
                    percentile_value = np.sum(metric_values < position_avg_percentiles[metric]) / len(metric_values)
                    position_percentiles.append(percentile_value)
    
                input_data = [grip_strength_bottom, grip_strength_top, vertical_jump, med_ball_situp, med_ball_chest]
                horsepower = vertical_jump + med_ball_situp + med_ball_chest
                input_data.append(horsepower)
    
                input_data_percentiles = []
                for i, metric in enumerate(all_metrics):
                    metric_values = pd.to_numeric(level_data[metric], errors='coerce').fillna(0)
                    percentile_value = np.sum(metric_values < input_data[i]) / len(metric_values)
                    input_data_percentiles.append(percentile_value)
    
                radar_img = plot_radar(input_data_percentiles, position_percentiles, all_metrics, player_name, f"{position} Average in {level}")
    
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(radar_img)
                with col2:
                    for i, metric in enumerate(all_metrics):
                        bar_img = plot_metric(metric, input_data_percentiles[i] * 100, input_data[i])
                        st.image(bar_img, use_column_width=True)
            else:
                st.error("No data found for the selected position and level.")
        else:
            st.warning("Please fill in all required player inputs.")

