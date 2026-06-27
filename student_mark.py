import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a mock dataset of student marks
data = {
    'StudentID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'Math': [85, 42, 78, 92, 55, 88, 35, 67, 95, 71],
    'Science': [90, 50, 82, 89, 60, 81, 45, 72, 98, 65],
    'English': [78, 62, 80, 85, 70, 75, 50, 68, 92, 74]
}

# Convert the dictionary into a Pandas DataFrame (Dataset)
df = pd.DataFrame(data)

# 2. Data Processing & Analytics
# Calculate Total Marks
df['Total'] = df['Math'] + df['Science'] + df['English']

# Calculate Average Percentage (assuming each subject is out of 100)
df['Percentage'] = (df['Total'] / 3).round(2)

# Determine Pass/Fail Status (Passing criteria: Min 40 in all subjects)
df['Status'] = np.where(
    (df['Math'] >= 40) & (df['Science'] >= 40) & (df['English'] >= 40), 
    'Pass', 
    'Fail'
)

# Assign Grades based on Percentage
def assign_grade(percentage):
    if percentage >= 90: return 'A+'
    elif percentage >= 80: return 'A'
    elif percentage >= 70: return 'B'
    elif percentage >= 50: return 'C'
    else: return 'F'

df['Grade'] = df['Percentage'].apply(assign_grade)

# 3. Display the Report Card Dataset
print("--- Student Marks Dataset ---")
print(df.to_string(index=False))
print("\n" + "="*50 + "\n")

# 4. Statistical Summary
print("--- Class Performance Summary ---")
summary = df[['Math', 'Science', 'English', 'Percentage']].mean().round(2)
print(f"Class Average in Math: {summary['Math']}")
print(f"Class Average in Science: {summary['Science']}")
print(f"Class Average in English: {summary['English']}")
print(f"Overall Class Average Percentage: {summary['Percentage']}%")

# Find the Topper
topper = df.loc[df['Total'].idxmax()]
print(f"Class Topper: {topper['Name']} with {topper['Total']} marks ({topper['Percentage']}%).")
print("\n" + "="*50 + "\n")

# 5. Visualizing the Data
print("Generating performance charts...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))

# Plotting Student Percentages
sns.barplot(x='Name', y='Percentage', data=df, hue='Status', palette={'Pass': 'green', 'Fail': 'red'})
plt.axhline(50, color='gray', linestyle='--', label='Passing Threshold (50%)')
plt.title('Student Overall Percentages')
plt.ylabel('Percentage (%)')
plt.xlabel('Student Name')
plt.legend()

plt.tight_layout()
plt.show()
