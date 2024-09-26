#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the penguins dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)

# Remove any rows with missing data for simplicity
penguins = penguins.dropna(subset=['species', 'flipper_length_mm'])

# Function to create the plot for each species
def create_histogram_with_markers(species_data, species_name):
    # Calculate stats
    mean_val = species_data['flipper_length_mm'].mean()
    median_val = species_data['flipper_length_mm'].median()
    min_val = species_data['flipper_length_mm'].min()
    max_val = species_data['flipper_length_mm'].max()
    q1 = species_data['flipper_length_mm'].quantile(0.25)
    q3 = species_data['flipper_length_mm'].quantile(0.75)
    std_val = species_data['flipper_length_mm'].std()
    
    lower_std_range = mean_val - 2 * std_val
    upper_std_range = mean_val + 2 * std_val

    # Create histogram
    fig = px.histogram(species_data, x='flipper_length_mm', nbins=20, title=f'Flipper Length Distribution for {species_name}')
    
    # Add vertical lines for mean and median
    fig.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top")
    fig.add_vline(x=median_val, line_width=2, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top")
    
    # Add shaded regions for range, IQR, and 2 standard deviations from the mean
    # Range (Min to Max)
    fig.add_vrect(x0=min_val, x1=max_val, fillcolor="lightgray", opacity=0.2, line_width=0, annotation_text="Range", annotation_position="top left")
    
    # Interquartile Range (Q1 to Q3)
    fig.add_vrect(x0=q1, x1=q3, fillcolor="lightblue", opacity=0.3, line_width=0, annotation_text="IQR", annotation_position="top left")
    
    # 2 Standard Deviations Range
    fig.add_vrect(x0=lower_std_range, x1=upper_std_range, fillcolor="lightgreen", opacity=0.2, line_width=0, annotation_text="±2σ", annotation_position="top left")
    
    # Show the plot
    fig.show()

# Separate the data by species and plot
species_list = penguins['species'].unique()
for species in species_list:
    species_data = penguins[penguins['species'] == species]
    create_histogram_with_markers(species_data, species)


# In[2]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the penguins dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)

# Remove any rows with missing data for simplicity
penguins = penguins.dropna(subset=['species', 'flipper_length_mm'])

# Create subplots: 1 row for each species, 3 columns for clear visibility
fig = make_subplots(rows=1, cols=3, subplot_titles=penguins['species'].unique())

# Function to add the histogram and statistical markers for each species to the subplot
def add_species_histogram_to_subplot(species_data, species_name, row, col):
    # Calculate stats
    mean_val = species_data['flipper_length_mm'].mean()
    median_val = species_data['flipper_length_mm'].median()
    min_val = species_data['flipper_length_mm'].min()
    max_val = species_data['flipper_length_mm'].max()
    q1 = species_data['flipper_length_mm'].quantile(0.25)
    q3 = species_data['flipper_length_mm'].quantile(0.75)
    std_val = species_data['flipper_length_mm'].std()

    lower_std_range = mean_val - 2 * std_val
    upper_std_range = mean_val + 2 * std_val

    # Create the histogram trace
    hist_trace = go.Histogram(x=species_data['flipper_length_mm'], nbinsx=20, name=f'{species_name} flipper_length_mm')

    # Add histogram to the subplot
    fig.add_trace(hist_trace, row=row, col=col)

    # Add vertical lines for mean and median
    fig.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top left", row=row, col=col)
    fig.add_vline(x=median_val, line_width=2, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top left", row=row, col=col)

    # Add shaded regions for range, IQR, and ±2 standard deviations from the mean
    # Range (Min to Max)
    fig.add_vrect(x0=min_val, x1=max_val, fillcolor="lightgray", opacity=0.2, line_width=0, row=row, col=col)

    # Interquartile Range (Q1 to Q3)
    fig.add_vrect(x0=q1, x1=q3, fillcolor="lightblue", opacity=0.3, line_width=0, row=row, col=col)

    # ±2 Standard Deviations Range
    fig.add_vrect(x0=lower_std_range, x1=upper_std_range, fillcolor="lightgreen", opacity=0.2, line_width=0, row=row, col=col)

# Iterate over the species and add each histogram to a different subplot
species_list = penguins['species'].unique()
for i, species in enumerate(species_list):
    species_data = penguins[penguins['species'] == species]
    add_species_histogram_to_subplot(species_data, species, row=1, col=i+1)

# Update layout
fig.update_layout(title_text="Flipper Length Distributions with Statistical Markers for Each Species",
                  height=600, width=1000, showlegend=False)

# Show the plot
fig.show()


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the penguins dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)

# Remove any rows with missing data for simplicity
penguins = penguins.dropna(subset=['species', 'flipper_length_mm'])

# Create subplots: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Function to plot KDE with statistical markers
def plot_kde_with_markers(ax, species_data, species_name):
    # Calculate stats
    mean_val = species_data['flipper_length_mm'].mean()
    median_val = species_data['flipper_length_mm'].median()
    min_val = species_data['flipper_length_mm'].min()
    max_val = species_data['flipper_length_mm'].max()
    q1 = species_data['flipper_length_mm'].quantile(0.25)
    q3 = species_data['flipper_length_mm'].quantile(0.75)
    std_val = species_data['flipper_length_mm'].std()

    lower_std_range = mean_val - 2 * std_val
    upper_std_range = mean_val + 2 * std_val

    # KDE plot
    sns.kdeplot(species_data['flipper_length_mm'], ax=ax, fill=True, label=species_name)
    ax.set_title(f'{species_name} Flipper Length KDE')

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='green', linestyle='--', label='Mean')
    ax.axvline(median_val, color='blue', linestyle='--', label='Median')

    # Add shaded regions
    # Range (Min to Max)
    ax.fill_betweenx(np.linspace(0, 0.03, 100), min_val, max_val, color='lightgray', alpha=0.3, label='Range')
    
    # Interquartile Range (Q1 to Q3)
    ax.fill_betweenx(np.linspace(0, 0.03, 100), q1, q3, color='lightblue', alpha=0.3, label='IQR')

    # ±2 Standard Deviations
    ax.fill_betweenx(np.linspace(0, 0.03, 100), lower_std_range, upper_std_range, color='lightgreen', alpha=0.2, label='±2σ')

    # Add legend
    ax.legend()

# Get unique species
species_list = penguins['species'].unique()

# Plot each species' KDE in its subplot
for i, species in enumerate(species_list):
    species_data = penguins[penguins['species'] == species]
    plot_kde_with_markers(axes[i], species_data, species)

# Adjust layout for clarity
plt.tight_layout()
plt.show()


# I prefer histograms because it visualizes the shape of the data. Histograms is very convenient when I need to see peaks and compare them with other data

# In[4]:


from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

n = 1500
data1 = stats.uniform.rvs(0, 10, size=n)
data2 = stats.norm.rvs(5, 1.5, size=n)
data3 = np.r_[stats.norm.rvs(2, 0.25, size=int(n/2)), stats.norm.rvs(8, 0.5, size=int(n/2))]
data4 = stats.norm.rvs(6, 0.5, size=n)

fig = make_subplots(rows=1, cols=4)

fig.add_trace(go.Histogram(x=data1, name='A', nbinsx=30, marker=dict(line=dict(color='black', width=1))), row=1, col=1)
fig.add_trace(go.Histogram(x=data2, name='B', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=2)
fig.add_trace(go.Histogram(x=data3, name='C', nbinsx=45, marker=dict(line=dict(color='black', width=1))), row=1, col=3)
fig.add_trace(go.Histogram(x=data4, name='D', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=4)

fig.update_layout(height=300, width=750, title_text="Row of Histograms")
fig.update_xaxes(title_text="A", row=1, col=1)
fig.update_xaxes(title_text="B", row=1, col=2)
fig.update_xaxes(title_text="C", row=1, col=3)
fig.update_xaxes(title_text="D", row=1, col=4)
fig.update_xaxes(range=[-0.5, 10.5])

for trace in fig.data:
    trace.xbins = dict(start=0, end=10)
    
# This code was produced by just making requests to Microsoft Copilot
# https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/COP/SLS/0001_concise_makeAplotV1.md

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[5]:


import numpy as np

# Calculate means and variances for each dataset
mean_data1 = np.mean(data1)
var_data1 = np.var(data1)

mean_data2 = np.mean(data2)
var_data2 = np.var(data2)

mean_data3 = np.mean(data3)
var_data3 = np.var(data3)

mean_data4 = np.mean(data4)
var_data4 = np.var(data4)

# Print the results
print(f"Data1 (Uniform): Mean = {mean_data1:.2f}, Variance = {var_data1:.2f}")
print(f"Data2 (Normal): Mean = {mean_data2:.2f}, Variance = {var_data2:.2f}")
print(f"Data3 (Bimodal): Mean = {mean_data3:.2f}, Variance = {var_data3:.2f}")
print(f"Data4 (Normal): Mean = {mean_data4:.2f}, Variance = {var_data4:.2f}")


# 4.1 data A and Data C
# 4.2 data A and Data B, data C and Data B
# 4.3 data A and Data C
# 4.4 Data C and Data D, Data A and Data D, Data B and Data D

# https://chatgpt.com/c/66f4d7bf-64ac-8000-829f-708cfa6b4965

# In[6]:


from scipy import stats
import pandas as pd
import numpy as np

# Generating a sample from a gamma distribution with shape parameter (a=2) and scale parameter (scale=2)
sample1 = stats.gamma(a=2, scale=2).rvs(size=1000)

# Plotting the histogram for the distribution
fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data")
# The .show(renderer="png") is a directive to display the plot in PNG format for specific submissions (GitHub/MarkUs)

# Calculating the mean of the sample
sample1.mean()

# Calculating the median of the sample using quantile function
np.quantile(sample1, [0.5]) # 0.5 corresponds to the median (50th percentile)

# Creating a second sample by negating the gamma distribution (this will have left skew)
sample2 = -stats.gamma(a=2, scale=2).rvs(size=1000)


# I believe that right skewness means the distribution leans towards the right, with the mean generally greater than the median. Left skewness means the distribution leans towards the left, and the mean is generally less than the median. I think this is influenced by some extreme maximum or minimum values, and it can also be affected by certain special circumstances.

# In[7]:


from scipy import stats
import pandas as pd
import numpy as np
import plotly.express as px

# Generate 1000 values from a right-skewed gamma distribution
sample1 = stats.gamma(a=2, scale=2).rvs(size=1000)

# Calculate the mean and median of the right-skewed distribution
mean1 = sample1.mean()
median1 = np.quantile(sample1, [0.5])[0]

# Display mean and median
print(f"Mean (Right Skewed): {mean1}")
print(f"Median (Right Skewed): {median1}")


# In[8]:


# Plot histogram for right-skewed distribution
fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data", title="Right-Skewed Distribution (Gamma)")
fig1.add_vline(x=mean1, line_width=3, line_dash="dash", line_color="red", annotation_text="Mean", annotation_position="top")
fig1.add_vline(x=median1, line_width=3, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top")

# Show the plot
fig1.show(renderer="png")


# In[9]:


# Generate 1000 values from a left-skewed (negated gamma) distribution
sample2 = -stats.gamma(a=2, scale=2).rvs(size=1000)

# Calculate the mean and median of the left-skewed distribution
mean2 = sample2.mean()
median2 = np.quantile(sample2, [0.5])[0]

# Display mean and median
print(f"Mean (Left Skewed): {mean2}")
print(f"Median (Left Skewed): {median2}")


# In[10]:


# Plot histogram for left-skewed distribution
fig2 = px.histogram(pd.DataFrame({'data': sample2}), x="data", title="Left-Skewed Distribution (Negated Gamma)")
fig2.add_vline(x=mean2, line_width=3, line_dash="dash", line_color="red", annotation_text="Mean", annotation_position="top")
fig2.add_vline(x=median2, line_width=3, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top")

# Show the plot
fig2.show(renderer="png")


# In[11]:


#### Explanation of Results:

- In the right-skewed distribution, we see that the **mean** is greater than the **median**. This occurs because the extreme values (the tail) on the right side pull the mean toward larger values.
- In the left-skewed distribution, the **mean** is less than the **median**. The extreme values on the left side (negative side) pull the mean toward smaller values.
- In both cases, skewness occurs because of extreme values that disproportionately influence the mean.


# In[12]:


# Generate 1000 values from a symmetric normal distribution
sample3 = np.random.normal(loc=0, scale=1, size=1000)

# Calculate the mean and median of the normal distribution
mean3 = sample3.mean()
median3 = np.quantile(sample3, [0.5])[0]

# Plot histogram for normal distribution
fig3 = px.histogram(pd.DataFrame({'data': sample3}), x="data", title="Symmetric Distribution (Normal)")
fig3.add_vline(x=mean3, line_width=3, line_dash="dash", line_color="red", annotation_text="Mean", annotation_position="top")
fig3.add_vline(x=median3, line_width=3, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top")

# Show the plot
fig3.show(renderer="png")


# In[13]:


import plotly.express as px
import pandas as pd

# For this example, we'll assume we want to see the distribution of shark attacks over time.
# Let's create a sample dataset (for example purposes) to simulate this.

# Assuming 'data' already has the 'Year' and 'some numeric column' for which you want to build histograms

# Create a sample dataset for illustration
# You can replace this with actual column names from your dataset
# For example, use the year and another relevant numeric column
sample_data = data[['Year', 'Sex ']].dropna()  # Replace 'Sex ' with the relevant numeric column

# Create an animated histogram
fig = px.histogram(sample_data, 
                   x="Year",  # Replace with the relevant column you want to histogram
                   color="Sex ",  # Replace with another relevant column, if desired
                   animation_frame="Year",  # Use the 'Year' for the animation frame
                   nbins=30,  # Adjust the number of bins as needed
                   title="Animated Histogram of Shark Attacks Over Time")

# Customize the layout if necessary
fig.update_layout(bargap=0.1)

# Display the animated histogram
fig.show()


# In[14]:


import pandas as pd
import plotly.express as px

# Load the dataset
file_path = 'path_to_your_jaws.csv'  # Adjust the file path accordingly
data = pd.read_csv(file_path, encoding='latin1')

# Cleaning the 'Year' column to ensure it contains valid numeric data
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Ensure 'Sex ' column is cleaned (whitespace removed and fill missing values)
data['Sex '] = data['Sex '].str.strip().fillna('Unknown')

# Drop rows with missing 'Year' or 'Sex ' values
sample_data = data[['Year', 'Sex ']].dropna()

# Create an animated histogram
fig = px.histogram(sample_data, 
                   x="Year",  
                   color="Sex ",  
                   animation_frame="Year",  
                   nbins=30,  
                   title="Animated Histogram of Shark Attacks Over Time")

# Customize the layout
fig.update_layout(bargap=0.1)

# Show the animated histogram
fig.show()


# In[15]:


import pandas as pd
import plotly.express as px

# Replace this with the correct path to your CSV file
file_path = 'path_to_your_jaws.csv'  # Adjust the file path accordingly

# Load the dataset
data = pd.read_csv(file_path, encoding='latin1')

# Cleaning the 'Year' column to ensure it contains valid numeric data
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Ensure 'Sex ' column is cleaned (whitespace removed and fill missing values)
data['Sex '] = data['Sex '].str.strip().fillna('Unknown')

# Drop rows with missing 'Year' or 'Sex ' values
sample_data = data[['Year', 'Sex ']].dropna()

# Create an animated histogram
fig = px.histogram(sample_data, 
                   x="Year",  
                   color="Sex ",  
                   animation_frame="Year",  
                   nbins=30,  
                   title="Animated Histogram of Shark Attacks Over Time")

# Customize the layout
fig.update_layout(bargap=0.1)

# Show the animated histogram
fig.show()


# In[16]:


import plotly.express as px
import pandas as pd

# Example dataset from Plotly (you can replace it with your own data)
df = px.data.gapminder()  # This is a default dataset from Plotly

# Create an animated histogram
fig = px.histogram(df, 
                   x="gdpPercap",  # Replace with the column you want on the x-axis
                   color="continent",  # Replace with the relevant column for coloring
                   animation_frame="year",  # The column to animate over (like 'Year')
                   nbins=30,  # Number of bins in the histogram
                   title="Animated Histogram of GDP per Capita Over Time")

# Customize the layout
fig.update_layout(bargap=0.1)

# Show the animated histogram
fig.show()


# In[17]:


9. Yes!


# 9. Yes

# In[ ]:




