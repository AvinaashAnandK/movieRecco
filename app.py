import streamlit as st
import pandas as pd
import ast
import plotly.express as px
import numpy as np
import glob
import os

@st.cache_data
def load_all_data():
    """
    Load and combine all CSV files
    """
    # List all CSV files in the data directory
    csv_files = glob.glob('data/*.csv')
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    dfs = []
    for idx, file in enumerate(csv_files):
        progress_text.text(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        dfs.append(df)
        # Update progress
        progress_bar.progress((idx + 1) / len(csv_files))
    
    progress_text.text("Combining datasets...")
    final_df = pd.concat(dfs, ignore_index=True)
    progress_bar.empty()
    progress_text.empty()
    
    return final_df

def get_data_type(value):
    try:
        evaluated = ast.literal_eval(str(value))
        return type(evaluated).__name__
    except:
        return type(value).__name__

def show_full_dataframe_analysis(df):
    st.header("Full Dataset Explorer")
    
    # Sidebar filters for the full dataframe
    with st.sidebar:
        st.header("Dataset Filters")
        
        # Search across all text columns
        search_term = st.text_input("Search across all text columns")
        
        # Multi-column selection
        selected_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=["title","media_type","vote_count","vote_average"]
        )
        
        # Numeric range filters for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            st.subheader("Numeric Filters")
            for col in numeric_columns:
                if col in selected_columns:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    selected_range = st.slider(
                        f"Filter {col}",
                        min_val, max_val,
                        (min_val, max_val)
                    )
                    df = df[df[col].between(selected_range[0], selected_range[1])]
        
        # Categorical filters
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            st.subheader("Categorical Filters")
            for col in categorical_columns:
                if col in selected_columns:
                    unique_values = df[col].unique()
                    if len(unique_values) < 50:  # Only show if not too many unique values
                        selected_cats = st.multiselect(
                            f"Filter {col}",
                            options=sorted(unique_values),
                            default=[]
                        )
                        if selected_cats:
                            df = df[df[col].isin(selected_cats)]

    # Main area display
    # Number of entries selector
    n_rows = st.number_input("Number of rows to display", min_value=1, max_value=len(df), value=min(10, len(df)))
    
    # Sort options
    sort_col = st.selectbox("Sort by", options=selected_columns)
    sort_ascending = st.checkbox("Sort ascending", True)
    df_display = df[selected_columns].sort_values(by=sort_col, ascending=sort_ascending)
    
    # Search functionality
    if search_term:
        mask = np.column_stack([df_display[col].astype(str).str.contains(search_term, case=False, na=False) 
                              for col in df_display.columns])
        df_display = df_display[mask.any(axis=1)]
    
    # Display dataframe with formatting
    st.dataframe(df_display.head(n_rows), use_container_width=True)
    
    # Download button
    st.download_button(
        label="Download filtered data as CSV",
        data=df_display.to_csv(index=False).encode('utf-8'),
        file_name='filtered_data.csv',
        mime='text/csv',
    )
    
    # Basic statistics
    if st.checkbox("Show summary statistics"):
        st.write(df_display.describe())

def show_column_analysis(df):
    st.header("Column Analysis")
    
    # Column selector
    selected_column = st.selectbox(
        "Select Column to Analyze",
        options=df.columns.tolist()
    )
    
    # Display column information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Column Info")
        sample_value = df[selected_column].iloc[0]
        data_type = get_data_type(sample_value)
        st.info(f"Data Type: {data_type}")
        st.metric("Non-null Count", df[selected_column].count())
        st.metric("Unique Values", df[selected_column].nunique())
    
    with col2:
        st.subheader("Sample Values")
        num_samples = st.slider("Number of samples", 1, 10, 5)
        st.write(df[selected_column].head(num_samples))
    
    # If numeric column, show distribution
    if np.issubdtype(df[selected_column].dtype, np.number):
        st.subheader("Distribution Plot")
        fig = px.histogram(df, x=selected_column)
        st.plotly_chart(fig, use_container_width=True)
    
    # If categorical column, show value counts
    elif df[selected_column].dtype == 'object':
        st.subheader("Value Counts")
        value_counts = df[selected_column].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values)
        st.plotly_chart(fig, use_container_width=True)

def show_visualization_tools(df):
    st.header("Visualization Tools")
    
    # Chart type selector
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Histogram"]
    )
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if chart_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", options=numeric_columns)
        y_col = st.selectbox("Y-axis", options=numeric_columns)
        color_col = st.selectbox("Color by", options=['None'] + df.columns.tolist())
        
        if color_col == 'None':
            fig = px.scatter(df, x=x_col, y=y_col)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Bar Chart":
        x_col = st.selectbox("X-axis", options=df.columns)
        y_col = st.selectbox("Y-axis", options=numeric_columns)
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Line Chart":
        x_col = st.selectbox("X-axis", options=df.columns)
        y_col = st.selectbox("Y-axis", options=numeric_columns)
        fig = px.line(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        x_col = st.selectbox("Groups", options=df.columns)
        y_col = st.selectbox("Values", options=numeric_columns)
        fig = px.box(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        col = st.selectbox("Column", options=numeric_columns)
        bins = st.slider("Number of bins", 5, 100, 30)
        fig = px.histogram(df, x=col, nbins=bins)
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Movie Dataset Analysis", layout="wide")
    
    st.title("Movie Dataset Analysis Tool")
    
    # Load data
    try:
        df = load_all_data()
        st.success(f"Successfully loaded dataset with {len(df)} rows")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Full Dataset", 
        "Column Analysis", 
        "Visualizations",
        "Search & Filter"
    ])
    
    with tab1:
        show_full_dataframe_analysis(df)
    
    with tab2:
        show_column_analysis(df)
    
    with tab3:
        show_visualization_tools(df)
    
    with tab4:
        st.header("Search & Filter")
        title_search = st.text_input("Search for a title")
        if title_search:
            filtered_df = df[df['title'].str.contains(title_search, case=False, na=False)]
            st.write(f"Found {len(filtered_df)} matches")
            st.dataframe(filtered_df)

if __name__ == "__main__":
    main()