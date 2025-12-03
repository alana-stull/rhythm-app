import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
df['Low_Productivity_Risk'] = np.where(df['productivity_0_100'] > 50, 1, 0)
Y = df['Low_Productivity_Risk']

# features
X_FEATURES_STRETCH = [
    'screen_time_hours',      
    'leisure_screen_hours',   
    'sleep_hours',            
    'exercise_minutes_per_week' # new for the stretch goal
]
X = df[X_FEATURES_STRETCH]

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# using decision tree
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, Y_train)

# making predictions
Y_pred_dt = model_dt.predict(X_test)

# evaluating model
accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
conf_matrix_dt = confusion_matrix(Y_test, Y_pred_dt)

print("\n--- Decision Tree Model Performance (Predicting Low Risk) ---")
print(f'Accuracy Score: {accuracy_dt:.4f}')
print("Confusion Matrix:")
print(conf_matrix_dt)

# decision tree application
importance_dt = pd.DataFrame({
    'Feature': X_FEATURES_STRETCH,
    'Importance': model_dt.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Decision Tree Feature Importance ---")
print(importance_dt)

# visualize the importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_dt, palette='plasma')
plt.title('Decision Tree Feature Importance (Low Productivity Risk)')
plt.show()