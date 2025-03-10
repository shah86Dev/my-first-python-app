import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Data Sweeper",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Data Sweeper")
st.markdown("### Upload, clean, and analyze your CSV and Excel files")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Upload Data", "Data Cleaning", "Data Analysis", "Data Visualization", "Export Data"]
page = st.sidebar.radio("Go to", pages)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

# Function to download dataframe as CSV
def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}_processed.csv">Download CSV File</a>'
    return href

# Function to download dataframe as Excel
def get_excel_download_link(df, filename):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}_processed.xlsx">Download Excel File</a>'
    return href

# Upload Data Page
if page == "Upload Data":
    st.header("Upload Your Data")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Get file extension
            file_extension = uploaded_file.name.split(".")[-1]
            
            # Read the file based on its extension
            if file_extension.lower() == "csv":
                data = pd.read_csv(uploaded_file)
            else:  # Excel file
                data = pd.read_excel(uploaded_file)
            
            # Store the data in session state
            st.session_state.data = data.copy()
            st.session_state.original_data = data.copy()
            st.session_state.filename = uploaded_file.name.split(".")[0]
            
            # Display success message and data preview
            st.success(f"File '{uploaded_file.name}' successfully loaded!")
            
            # Display data info
            st.subheader("Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {data.shape[0]}")
                st.write(f"**Columns:** {data.shape[1]}")
            with col2:
                st.write(f"**Missing values:** {data.isna().sum().sum()}")
                st.write(f"**Duplicate rows:** {data.duplicated().sum()}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            # Display column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes,
                'Missing Values': data.isna().sum().values,
                'Unique Values': [data[col].nunique() for col in data.columns]
            })
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV or Excel file to get started.")
        
        # Sample data option
        if st.button("Use Sample Data"):
            # Create sample data
            sample_data = pd.DataFrame({
                'ID': range(1, 101),
                'Name': [f"Person {i}" for i in range(1, 101)],
                'Age': np.random.randint(18, 65, 100),
                'Salary': np.random.randint(30000, 100000, 100),
                'Department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Sales'], 100),
                'Years': np.random.randint(1, 20, 100),
                'Rating': np.random.uniform(1, 5, 100).round(1)
            })
            
            # Add some missing values
            sample_data.loc[np.random.choice(sample_data.index, 10), 'Age'] = np.nan
            sample_data.loc[np.random.choice(sample_data.index, 10), 'Salary'] = np.nan
            
            # Add some duplicates
            sample_data.loc[90:94] = sample_data.loc[0:4].values
            
            # Store the data in session state
            st.session_state.data = sample_data.copy()
            st.session_state.original_data = sample_data.copy()
            st.session_state.filename = "sample_data"
            
            st.success("Sample data loaded successfully!")
            st.dataframe(sample_data.head(10))

# Data Cleaning Page
elif page == "Data Cleaning":
    st.header("Data Cleaning")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Create tabs for different cleaning operations
        cleaning_tabs = st.tabs(["Missing Values", "Duplicates", "Data Types", "Filtering", "Reset Data"])
        
        # Missing Values tab
        with cleaning_tabs[0]:
            st.subheader("Handle Missing Values")
            
            # Display columns with missing values
            missing_cols = data.columns[data.isna().any()].tolist()
            
            if missing_cols:
                st.write(f"Columns with missing values: {', '.join(missing_cols)}")
                
                # Select column to handle missing values
                col_to_clean = st.selectbox("Select column to handle missing values:", missing_cols)
                
                # Select method to handle missing values
                missing_method = st.radio(
                    "Select method to handle missing values:",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"]
                )
                
                if st.button("Apply Missing Value Handling"):
                    if missing_method == "Drop rows":
                        st.session_state.data = data.dropna(subset=[col_to_clean])
                        st.success(f"Dropped rows with missing values in column '{col_to_clean}'")
                    
                    elif missing_method == "Fill with mean":
                        if pd.api.types.is_numeric_dtype(data[col_to_clean]):
                            st.session_state.data[col_to_clean] = data[col_to_clean].fillna(data[col_to_clean].mean())
                            st.success(f"Filled missing values in '{col_to_clean}' with mean")
                        else:
                            st.error("Mean can only be calculated for numeric columns")
                    
                    elif missing_method == "Fill with median":
                        if pd.api.types.is_numeric_dtype(data[col_to_clean]):
                            st.session_state.data[col_to_clean] = data[col_to_clean].fillna(data[col_to_clean].median())
                            st.success(f"Filled missing values in '{col_to_clean}' with median")
                        else:
                            st.error("Median can only be calculated for numeric columns")
                    
                    elif missing_method == "Fill with mode":
                        mode_value = data[col_to_clean].mode()[0]
                        st.session_state.data[col_to_clean] = data[col_to_clean].fillna(mode_value)
                        st.success(f"Filled missing values in '{col_to_clean}' with mode")
                    
                    elif missing_method == "Fill with custom value":
                        custom_value = st.text_input("Enter custom value:")
                        if custom_value:
                            handle_custom_value(col_to_clean, custom_value, data)
                
                # Update data reference
                data = st.session_state.data
                
                # Display updated missing values count
                st.write(f"Current missing values count: {data.isna().sum().sum()}")
            else:
                st.write("No missing values found in the dataset.")
        
        # Duplicates tab
        with cleaning_tabs[1]:
            st.subheader("Handle Duplicates")
            
            # Check for duplicates
            duplicate_count = data.duplicated().sum()
            
            if duplicate_count > 0:
                st.write(f"Number of duplicate rows: {duplicate_count}")
                
                # Select columns to consider for duplicates
                all_columns = data.columns.tolist()
                cols_for_duplicates = st.multiselect(
                    "Select columns to consider for identifying duplicates (leave empty for all columns):",
                    all_columns
                )
                
                if st.button("Remove Duplicates"):
                    if cols_for_duplicates:
                        st.session_state.data = data.drop_duplicates(subset=cols_for_duplicates)
                        st.success(f"Removed duplicates based on selected columns. {len(data) - len(st.session_state.data)} rows removed.")
                    else:
                        st.session_state.data = data.drop_duplicates()
                        st.success(f"Removed all duplicate rows. {len(data) - len(st.session_state.data)} rows removed.")
                
                # Update data reference
                data = st.session_state.data
                
                # Display updated duplicate count
                st.write(f"Current duplicate rows count: {data.duplicated().sum()}")
            else:
                st.write("No duplicate rows found in the dataset.")
        
        # Data Types tab
        with cleaning_tabs[2]:
            st.subheader("Change Data Types")
            
            # Select column to change data type
            col_to_change = st.selectbox("Select column to change data type:", data.columns)
            
            # Display current data type
            current_type = data[col_to_change].dtype
            st.write(f"Current data type: {current_type}")
            
            # Select new data type
            new_type = st.selectbox(
                "Select new data type:",
                ["int", "float", "str", "datetime", "category"]
            )
            
            if st.button("Change Data Type"):
                try:
                    if new_type == "int":
                        st.session_state.data[col_to_change] = data[col_to_change].astype(int)
                    elif new_type == "float":
                        st.session_state.data[col_to_change] = data[col_to_change].astype(float)
                    elif new_type == "str":
                        st.session_state.data[col_to_change] = data[col_to_change].astype(str)
                    elif new_type == "datetime":
                        st.session_state.data[col_to_change] = pd.to_datetime(data[col_to_change])
                    elif new_type == "category":
                        st.session_state.data[col_to_change] = data[col_to_change].astype('category')
                    
                    st.success(f"Changed data type of '{col_to_change}' to {new_type}")
                except Exception as e:
                    st.error(f"Error changing data type: {e}")
        
        # Filtering tab
        with cleaning_tabs[3]:
            st.subheader("Filter Data")
            
            # Select column to filter
            filter_col = st.selectbox("Select column to filter:", data.columns)
            
            # Different filter options based on column data type
            if pd.api.types.is_numeric_dtype(data[filter_col]):
                min_val = float(data[filter_col].min())
                max_val = float(data[filter_col].max())
                
                filter_range = st.slider(
                    f"Filter range for {filter_col}",
                    min_val, max_val, (min_val, max_val)
                )
                
                if st.button("Apply Numeric Filter"):
                    filtered_data = data[(data[filter_col] >= filter_range[0]) & (data[filter_col] <= filter_range[1])]
                    st.session_state.data = filtered_data
                    st.success(f"Data filtered. {len(filtered_data)} rows remaining.")
            
            else:  # Categorical or text data
                unique_values = data[filter_col].dropna().unique().tolist()
                selected_values = st.multiselect(
                    f"Select values to keep from {filter_col}",
                    unique_values,
                    default=unique_values
                )
                
                if st.button("Apply Categorical Filter"):
                    if selected_values:
                        filtered_data = data[data[filter_col].isin(selected_values)]
                        st.session_state.data = filtered_data
                        st.success(f"Data filtered. {len(filtered_data)} rows remaining.")
                    else:
                        st.warning("No values selected. No filtering applied.")
        
        # Reset Data tab
        with cleaning_tabs[4]:
            st.subheader("Reset Data")
            
            if st.button("Reset to Original Data"):
                st.session_state.data = st.session_state.original_data.copy()
                st.success("Data reset to original state.")
        
        # Display current data preview
        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.data.head(10))
        st.write(f"Current shape: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
    
    else:
        st.warning("Please upload data in the 'Upload Data' section first.")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Create tabs for different analysis operations
        analysis_tabs = st.tabs(["Summary Statistics", "Correlation Analysis", "Group Analysis"])
        
        # Summary Statistics tab
        with analysis_tabs[0]:
            st.subheader("Summary Statistics")
            
            # Select columns for summary statistics
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect(
                    "Select columns for summary statistics:",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if selected_cols:
                    st.write(data[selected_cols].describe())
                else:
                    st.info("Please select at least one column.")
            else:
                st.info("No numeric columns found in the dataset.")
        
        # Correlation Analysis tab
        with analysis_tabs[1]:
            st.subheader("Correlation Analysis")
            
            # Select columns for correlation analysis
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select columns for correlation analysis:",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    corr_matrix = data[selected_cols].corr()
                    
                    # Display correlation matrix
                    st.write("Correlation Matrix:")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                    
                    # Plot correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr_matrix, cmap='coolwarm')
                    
                    # Add colorbar
                    plt.colorbar(im)
                    
                    # Add ticks and labels
                    ax.set_xticks(np.arange(len(selected_cols)))
                    ax.set_yticks(np.arange(len(selected_cols)))
                    ax.set_xticklabels(selected_cols, rotation=45, ha='right')
                    ax.set_yticklabels(selected_cols)
                    
                    # Add correlation values
                    for i in range(len(selected_cols)):
                        for j in range(len(selected_cols)):
                            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                          ha="center", va="center", color="black")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Please select at least two columns.")
            else:
                st.info("Need at least two numeric columns for correlation analysis.")
        
        # Group Analysis tab
        with analysis_tabs[2]:
            st.subheader("Group Analysis")
            
            # Select groupby column
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                group_col = st.selectbox("Select column to group by:", categorical_cols)
                
                # Select columns to aggregate
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    agg_cols = st.multiselect(
                        "Select columns to aggregate:",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))]
                    )
                    
                    # Select aggregation functions
                    agg_funcs = st.multiselect(
                        "Select aggregation functions:",
                        ["mean", "median", "sum", "min", "max", "count", "std"],
                        default=["mean", "sum", "count"]
                    )
                    
                    if agg_cols and agg_funcs:
                        # Create aggregation dictionary
                        agg_dict = {col: agg_funcs for col in agg_cols}
                        
                        # Perform groupby operation
                        grouped_data = data.groupby(group_col).agg(agg_dict)
                        
                        # Display grouped data
                        st.write("Grouped Data:")
                        st.dataframe(grouped_data)
                        
                        # Option to download grouped data
                        if st.button("Download Grouped Data"):
                            csv = grouped_data.to_csv()
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="grouped_data.csv">Download Grouped Data CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.info("Please select at least one column to aggregate and one aggregation function.")
                else:
                    st.info("No numeric columns found for aggregation.")
            else:
                st.info("No categorical columns found for grouping.")
    
    else:
        st.warning("Please upload data in the 'Upload Data' section first.")

# Data Visualization Page
elif page == "Data Visualization":
    st.header("Data Visualization")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["Histogram", "Scatter Plot", "Bar Chart", "Box Plot", "Line Chart"])
        
        # Histogram tab
        with viz_tabs[0]:
            st.subheader("Histogram")
            
            # Select column for histogram
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                hist_col = st.selectbox("Select column for histogram:", numeric_cols, key="hist_col")
                
                # Number of bins
                bins = st.slider("Number of bins:", 5, 100, 20, key="hist_bins")
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(data[hist_col].dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add labels and title
                ax.set_xlabel(hist_col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {hist_col}")
                
                # Display plot
                st.pyplot(fig)
            else:
                st.info("No numeric columns found for histogram.")
        
        # Scatter Plot tab
        with viz_tabs[1]:
            st.subheader("Scatter Plot")
            
            # Select columns for scatter plot
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X-axis column:", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Select Y-axis column:", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
                
                # Optional color column
                color_option = st.checkbox("Add color dimension", key="scatter_color_option")
                
                if color_option:
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    if categorical_cols:
                        color_col = st.selectbox("Select column for color:", categorical_cols, key="scatter_color")
                        
                        # Create scatter plot with color
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Get unique categories
                        categories = data[color_col].unique()
                        
                        # Plot each category
                        for category in categories:
                            subset = data[data[color_col] == category]
                            ax.scatter(subset[x_col], subset[y_col], label=category, alpha=0.7)
                        
                        # Add legend
                        ax.legend()
                    else:
                        st.info("No categorical columns found for color dimension.")
                        
                        # Create simple scatter plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(data[x_col], data[y_col], alpha=0.7, color='skyblue')
                else:
                    # Create simple scatter plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(data[x_col], data[y_col], alpha=0.7, color='skyblue')
                
                # Add labels and title
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
                
                # Display plot
                st.pyplot(fig)
            else:
                st.info("Need at least two numeric columns for scatter plot.")
        
        # Bar Chart tab
        with viz_tabs[2]:
            st.subheader("Bar Chart")
            
            # Select columns for bar chart
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                x_col = st.selectbox("Select category column (X-axis):", categorical_cols, key="bar_x")
                
                # Select value column
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    y_col = st.selectbox("Select value column (Y-axis):", numeric_cols, key="bar_y")
                    
                    # Select aggregation function
                    agg_func = st.selectbox(
                        "Select aggregation function:",
                        ["count", "sum", "mean", "median", "min", "max"],
                        key="bar_agg"
                    )
                    
                    # Prepare data for bar chart
                    if agg_func == "count":
                        bar_data = data[x_col].value_counts()
                        title = f"Count of {x_col}"
                        y_label = "Count"
                    else:
                        # Group by the categorical column and aggregate
                        if agg_func == "mean":
                            bar_data = data.groupby(x_col)[y_col].mean()
                        elif agg_func == "median":
                            bar_data = data.groupby(x_col)[y_col].median()
                        elif agg_func == "sum":
                            bar_data = data.groupby(x_col)[y_col].sum()
                        elif agg_func == "min":
                            bar_data = data.groupby(x_col)[y_col].min()
                        elif agg_func == "max":
                            bar_data = data.groupby(x_col)[y_col].max()
                        
                        title = f"{agg_func.capitalize()} of {y_col} by {x_col}"
                        y_label = f"{agg_func.capitalize()} of {y_col}"
                    
                    # Limit number of categories if too many
                    if len(bar_data) > 20:
                        st.warning(f"Too many categories ({len(bar_data)}). Showing top 20.")
                        bar_data = bar_data.nlargest(20)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bar_data.plot(kind='bar', ax=ax, color='skyblue')
                    
                    # Add labels and title
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_label)
                    ax.set_title(title)
                    
                    # Rotate x-axis labels
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    
                    # Display plot
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns found for bar chart values.")
            else:
                st.info("No categorical columns found for bar chart categories.")
        
        # Box Plot tab
        with viz_tabs[3]:
            st.subheader("Box Plot")
            
            # Select numeric column for box plot
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                y_col = st.selectbox("Select numeric column for box plot:", numeric_cols, key="box_y")
                
                # Optional grouping
                group_option = st.checkbox("Group by category", key="box_group_option")
                
                if group_option:
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if categorical_cols:
                        x_col = st.selectbox("Select grouping column:", categorical_cols, key="box_x")
                        
                        # Limit number of categories if too many
                        categories = data[x_col].unique()
                        if len(categories) > 10:
                            st.warning(f"Too many categories ({len(categories)}). Consider filtering your data.")
                            
                            # Get top categories by count
                            top_cats = data[x_col].value_counts().nlargest(10).index.tolist()
                            use_top = st.checkbox("Use top 10 categories by frequency", value=True, key="box_top_cats")
                            
                            if use_top:
                                plot_data = data[data[x_col].isin(top_cats)]
                            else:
                                plot_data = data
                        else:
                            plot_data = data
                        
                        # Create grouped box plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.boxplot([plot_data[plot_data[x_col] == cat][y_col].dropna() for cat in plot_data[x_col].unique()],
                                  labels=plot_data[x_col].unique())
                        
                        # Add labels and title
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"Box Plot of {y_col} by {x_col}")
                        
                        # Rotate x-axis labels if needed
                        plt.xticks(rotation=45, ha='right')
                    else:
                        st.info("No categorical columns found for grouping.")
                        
                        # Create simple box plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.boxplot(data[y_col].dropna())
                        ax.set_ylabel(y_col)
                        ax.set_title(f"Box Plot of {y_col}")
                        ax.set_xticks([1])
                        ax.set_xticklabels([y_col])
                else:
                    # Create simple box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.boxplot(data[y_col].dropna())
                    ax.set_ylabel(y_col)
                    ax.set_title(f"Box Plot of {y_col}")
                    ax.set_xticks([1])
                    ax.set_xticklabels([y_col])
                
                # Display plot
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No numeric columns found for box plot.")
        
        # Line Chart tab
        with viz_tabs[4]:
            st.subheader("Line Chart")
            
            # Check if there are any datetime columns
            datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            if not datetime_cols:
                st.info("No datetime columns found. You can convert a column to datetime in the 'Data Cleaning' section.")
                
                # Allow using a numeric column as X-axis
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column (numeric):", numeric_cols, key="line_x")
                    y_col = st.selectbox("Select Y-axis column:", numeric_cols, key="line_y", index=min(1, len(numeric_cols)-1))
                    
                    # Sort data by x_col
                    plot_data = data.sort_values(by=x_col)
                    
                    # Create line chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(plot_data[x_col], plot_data[y_col], marker='o', linestyle='-', color='blue')
                    
                    # Add labels and title
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"Line Chart: {y_col} vs {x_col}")
                    
                    # Display plot
                    st.pyplot(fig)
                else:
                    st.info("Need at least two numeric columns for line chart.")
            else:
                # Use datetime column as X-axis
                x_col = st.selectbox("Select datetime column (X-axis):", datetime_cols, key="line_datetime_x")
                
                # Select Y-axis column(s)
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    y_cols = st.multiselect("Select Y-axis column(s):", numeric_cols, default=[numeric_cols[0]], key="line_y_multi")
                    
                    if y_cols:
                        # Sort data by datetime
                        plot_data = data.sort_values(by=x_col)
                        
                        # Create line chart
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for y_col in y_cols:
                            ax.plot(plot_data[x_col], plot_data[y_col], marker='o', linestyle='-', label=y_col)
                        
                        # Add legend if multiple columns
                        if len(y_cols) > 1:
                            ax.legend()
                        
                        # Add labels and title
                        ax.set_xlabel(x_col)
                        ax.set_ylabel("Value")
                        ax.set_title(f"Line Chart by {x_col}")
                        
                        # Format x-axis as dates
                        fig.autofmt_xdate()
                        
                        # Display plot
                        st.pyplot(fig)
                    else:
                        st.info("Please select at least one Y-axis column.")
                else:
                    st.info("No numeric columns found for Y-axis values.")
    
    else:
        st.warning("Please upload data in the 'Upload Data' section first.")

# Export Data Page
elif page == "Export Data":
    st.header("Export Processed Data")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        
        # Display data info
        st.subheader("Data Information")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        # Export options
        st.subheader("Export Options")
        
        # Select columns to export
        all_columns = data.columns.tolist()
        export_cols = st.multiselect(
            "Select columns to export (leave empty to export all columns):",
            all_columns,
            default=all_columns
        )
        
        # Export format
        export_format = st.radio("Select export format:", ["CSV", "Excel"])
        
        # Export button
        if st.button("Export Data"):
            if export_cols:
                export_data = data[export_cols]
            else:
                export_data = data
            
            filename = st.session_state.filename or "exported_data"
            
            if export_format == "CSV":
                st.markdown(get_csv_download_link(export_data, filename), unsafe_allow_html=True)
                st.success("CSV file ready for download!")
            else:  # Excel
                st.markdown(get_excel_download_link(export_data, filename), unsafe_allow_html=True)
                st.success("Excel file ready for download!")
        
        # Compare with original data
        if st.checkbox("Show comparison with original data"):
            if st.session_state.original_data is not None:
                original_data = st.session_state.original_data
                
                st.subheader("Comparison with Original Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Data**")
                    st.write(f"Rows: {original_data.shape[0]}, Columns: {original_data.shape[1]}")
                    st.write(f"Missing values: {original_data.isna().sum().sum()}")
                    st.write(f"Duplicate rows: {original_data.duplicated().sum()}")
                
                with col2:
                    st.write("**Processed Data**")
                    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
                    st.write(f"Missing values: {data.isna().sum().sum()}")
                    st.write(f"Duplicate rows: {data.duplicated().sum()}")
                
                # Calculate changes
                rows_diff = data.shape[0] - original_data.shape[0]
                cols_diff = data.shape[1] - original_data.shape[1]
                missing_diff = data.isna().sum().sum() - original_data.isna().sum().sum()
                dupes_diff = data.duplicated().sum() - original_data.duplicated().sum()
                
                st.subheader("Changes Made")
                st.write(f"Rows: {'Added' if rows_diff > 0 else 'Removed'} {abs(rows_diff)} rows" if rows_diff != 0 else "No change in row count")
                st.write(f"Columns: {'Added' if cols_diff > 0 else 'Removed'} {abs(cols_diff)} columns" if cols_diff != 0 else "No change in column count")
                st.write(f"Missing values: {'Added' if missing_diff > 0 else 'Removed'} {abs(missing_diff)} missing values" if missing_diff != 0 else "No change in missing values")
                st.write(f"Duplicate rows: {'Added' if dupes_diff > 0 else 'Removed'} {abs(dupes_diff)} duplicate rows" if dupes_diff != 0 else "No change in duplicate rows")
    
    else:
        st.warning("Please upload data in the 'Upload Data' section first.")

# Run the app
if __name__ == "__main__":
    # This is already handled by Streamlit
    pass

def handle_custom_value(col_to_clean, custom_value, data):
    try:
        if pd.api.types.is_numeric_dtype(data[col_to_clean]):
            custom_value = float(custom_value)
    except ValueError:
        st.error("Please enter a numeric value")
        return
        
    st.session_state.data[col_to_clean] = data[col_to_clean].fillna(custom_value)
    st.success(f"Filled missing values in '{col_to_clean}' with '{custom_value}'")