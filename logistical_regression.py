import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load data
csv_filename = 'ScreenTime vs MentalWellness.csv'
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: Could not find '{csv_filename}'. Ensure it is in the same directory.")
    exit()

# low productivity risk variable
# 1 = low risk (productivity > 50), 0 = high risk (productivity <= 50)
df['Low_Productivity_Risk'] = np.where(df['productivity_0_100'] > 50, 1, 0)
Y = df['Low_Productivity_Risk']

# features
X_FEATURES = [
    'screen_time_hours',      # overall digital load (work and leisure)
    'leisure_screen_hours',   # specific digital habit
    'sleep_hours',            # foundational wellness anchor
]
X = df[X_FEATURES]

print(f"Features Selected: {X_FEATURES}")

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# logistic regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, Y_train)

# making predictions
Y_pred = model.predict(X_test)

# evaluating model
# used accuracy and confusion matrix instead of MSE/R2. (gemini suggested bc of logistic regression instead of linear)
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("\n--- Logistic Regression Model Performance ---")
print(f'Accuracy Score: {accuracy:.4f}')
print("Confusion Matrix:")
print(conf_matrix)

# visualization of results
coefficients = pd.DataFrame({
    'Feature': X_FEATURES,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\n--- Model Coefficients (Feature Importance) ---")
print(coefficients)

# visualize coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='coolwarm')
plt.title('Feature Coefficients Predicting Low Risk (Rhythm Flow State)')
plt.axvline(0, color='black', linestyle='--') # line at 0 helps distinguish positive vs. negative influence
plt.show()
