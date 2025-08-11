# EDA-Titanic-cleaned-data
Titanic dataset analysis using Python and Pandas. Covers data preprocessing, statistical summaries, and visual exploration to understand survival trends.
# Titanic Dataset - Exploratory Data Analysis (EDA)

## 📌 Objective
The goal of this project is to perform data cleaning and exploratory data analysis (EDA) on the Titanic dataset to uncover patterns and trends related to passenger survival.

## 📂 Dataset
- **File**: `train.csv`
- **Rows**: 891
- **Columns**: 12
- **Key Features**:
  - PassengerId: Unique ID
  - Survived: Target (0 = No, 1 = Yes)
  - Pclass: Passenger class
  - Name, Sex, Age
  - SibSp, Parch: Family onboard
  - Ticket, Fare
  - Cabin, Embarked

## ⚙️ Requirements
Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn

 How to Run
Clone the repo:

bash
Copy code
git clone https://github.com/your-username/titanic-eda.git
Open titanic_eda.ipynb in Jupyter Notebook or VS Code.

Run all cells to see cleaning and visualizations.

🧹 Data Cleaning Steps
Checked for missing values

Filled missing Age with median

Filled missing Embarked with mode

Dropped Cabin due to high missing percentage

Converted Sex to numeric for analysis

📊 EDA Highlights
Women had a higher survival rate than men

Higher-class passengers had better survival rates

Younger passengers tended to survive more often

📝 Insights
Passenger class strongly influenced survival.

Gender was a key factor — females more likely to survive.

Age had a correlation with survival — children prioritized.

📌 Recommendations
In rescue scenarios, prioritize women, children, and higher-class cabins.

Additional data could improve model accuracy for prediction tasks.

pgsql
Copy code

---

## **titanic_eda.ipynb** (Code)
```python
# Titanic EDA & Data Cleaning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Basic info
print(df.head())
print(df.info())
print(df.describe())

# Missing values check
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to many missing values
df.drop(columns=['Cabin'], inplace=True)

# Convert 'Sex' to numeric for analysis
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Verify no missing values remain
print("\nMissing Values after cleaning:\n", df.isnull().sum())

# EDA: Survival by gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Count by Gender')
plt.show()

# EDA: Survival by class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.show()

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival')
plt.show()

