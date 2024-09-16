import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")

# Title
st.title('Automated Data Analysis')
st.text('Student attendance analysis')

# Function for EDA
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')

def data_summary():
    st.header('Statistics of Dataframe')
    st.write(df.describe(include='all'))

def data_shape():
    st.header('Show Shape')
    st.write(df.shape)

def data_head():
    st.header('Show Head')
    st.write(df.head())

def data_tail():
    st.header('Show Tail')
    st.write(df.tail())

# Function for plotting
def plot_histogram(column):
    plt.figure(figsize=(10, 5))
    plt.hist(df[column].dropna(), bins=30, edgecolor='k')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)

def plot_bar_chart(column):
    plt.figure(figsize=(10, 5))
    df[column].value_counts().plot(kind='bar', edgecolor='k')
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    st.pyplot(plt)

def plot_pie_chart(column):
    plt.figure(figsize=(10, 5))
    df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Pie Chart of {column}')
    st.pyplot(plt)

def plot_line_chart(column):
    plt.figure(figsize=(10, 5))
    plt.plot(df[column].dropna(), marker='o')
    plt.title(f'Line Chart of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    st.pyplot(plt)

# Sidebar
st.sidebar.title('Sidebar')

# File uploader with support for CSV, Excel, and Text files
upload_file = st.sidebar.file_uploader('Upload a file (CSV, Excel, or TXT)', type=['csv', 'xlsx', 'xls', 'txt'])

# Reading the uploaded file
if upload_file is not None:
    file_type = upload_file.name.split('.')[-1]
    
    if file_type == 'csv':
        df = pd.read_csv(upload_file)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(upload_file)
    elif file_type == 'txt':
        df = pd.read_csv(upload_file, delimiter='\t')

    # Sidebar navigation
    st.sidebar.title('EDA Options')
    show_home = st.sidebar.checkbox('Home')
    show_summary = st.sidebar.checkbox('Data Summary')
    show_shape = st.sidebar.checkbox('Data Shape')
    show_head = st.sidebar.checkbox('Data Head')
    show_tail = st.sidebar.checkbox('Data Tail')

    # Display options based on checkboxes
    if show_home:
        home(upload_file)
    if show_summary:
        data_summary()
    if show_shape:
        data_shape()
    if show_head:
        data_head()
    if show_tail:
        data_tail()

    # Sidebar plotting options
    st.sidebar.title('Plot Options')
    plot_histogram_option = st.sidebar.checkbox('Plot Histogram')
    plot_bar_chart_option = st.sidebar.checkbox('Plot Bar Chart')
    plot_pie_chart_option = st.sidebar.checkbox('Plot Pie Chart')
    plot_line_chart_option = st.sidebar.checkbox('Plot Line Chart')

    # Plotting based on checkboxes
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns

    if plot_histogram_option and len(numerical_cols) > 0:
        selected_col = st.sidebar.selectbox('Select a column for histogram', numerical_cols)
        plot_histogram(selected_col)
    
    if plot_bar_chart_option and len(categorical_cols) > 0:
        selected_col = st.sidebar.selectbox('Select a column for bar chart', categorical_cols)
        plot_bar_chart(selected_col)

    if plot_pie_chart_option and len(categorical_cols) > 0:
        selected_col = st.sidebar.selectbox('Select a column for pie chart', categorical_cols)
        plot_pie_chart(selected_col)
    
    if plot_line_chart_option and len(numerical_cols) > 0:
        selected_col = st.sidebar.selectbox('Select a column for line chart', numerical_cols)
        plot_line_chart(selected_col)

    # Function to add a total row to the DataFrame
    def add_total_row(df):
        df_sum = df.select_dtypes(include=[np.number]).fillna(0).sum()
        sum_df = pd.DataFrame(df_sum).T
        sum_df.index = ['Total']  
        df_with_total = pd.concat([df, sum_df])
        return df_with_total

    # Function to add a total column to the DataFrame
    def add_total_column(df):
        df['Row_Total'] = df.select_dtypes(include=[np.number]).fillna(0).sum(axis=1)
        return df

    # Function to calculate percentage for numerical rows
    def add_row_percentage(df):
        numerical_df = df.select_dtypes(include=[np.number])
        df_percentage = numerical_df.div(df['Row_Total'], axis=0) * 100
        df_percentage = df_percentage.fillna(0)
        return df_percentage.add_suffix('_Row%')

    # Function to generate a combined DataFrame
    def generate_combined_df(df):
        df_with_total = add_total_row(df)
        df_with_total_column = add_total_column(df_with_total)
        df_with_row_percentage = add_row_percentage(df_with_total_column)

        combined_df = df_with_total_column.copy()
        combined_df = pd.concat([combined_df, df_with_row_percentage], axis=1)
        
        return combined_df

    # Add a checkbox to show DataFrame with all totals and percentages
    show_all_totals_and_percentages = st.sidebar.checkbox('Show All Totals and Percentages')

    if show_all_totals_and_percentages:
        st.header('DataFrame with Totals and Percentages')
        combined_df = generate_combined_df(df)
        st.write(combined_df)

else:
    st.sidebar.write('Please upload a file to begin.')
