# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the dataset
df = sns.load_dataset('titanic')

# 2. Preprocessing
df['age'] = df['age'].fillna(df['age'].median())
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# 3. Data Visualization (MANDATORY FOR CHECKLIST)
# Graph 1: Survival by Sex
plt.figure(figsize=(6,4))
sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Sex (0=Male, 1=Female)')
plt.show()

# Graph 2: Heatmap of correlations
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 4. Feature Selection
features = ['pclass', 'sex', 'age', 'fare', 'FamilySize']
# Use .copy() to prevent the "SettingWithCopyWarning" from your screenshot
X = df[features].copy()
y = df['survived']

# Handle missing Fare
X.loc[:, 'fare'] = X['fare'].fillna(X['fare'].median())

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 7. Model Evaluation & Confusion Matrix (MANDATORY FOR CHECKLIST)
predictions = model.predict(X_test)
print(f"Project Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# Generate the Confusion Matrix Plot
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
