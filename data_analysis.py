import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading data + summary statistics
csv_filename = 'ScreenTime vs MentalWellness.csv'
try:
    df = pd.read_csv(csv_filename)
    print(f"Dataset '{csv_filename}' loaded successfully!\n")
except FileNotFoundError:
    print(f"Error: Could not find '{csv_filename}'. Make sure the file is in the same directory.")
    exit()

print("--- First 5 Rows of Data (for column inspection) ---")
print(df.head())

print("\n--- Summary Statistics (Mean, Std Dev, Min/Max) ---")
print(df.describe())

# exploratory data visualization
X_FEATURE = 'leisure_screen_hours'
Y_TARGET = 'stress_level_0_10'

# check for column existence before plotting
if X_FEATURE in df.columns and Y_TARGET in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_FEATURE, y=Y_TARGET, data=df)

    # calculate and display correlation for context
    correlation = df[X_FEATURE].corr(df[Y_TARGET])
    plt.title(f'Relationship: {X_FEATURE} vs. {Y_TARGET} (Correlation: {correlation:.2f})')
    plt.xlabel(X_FEATURE)
    plt.ylabel(Y_TARGET)
    plt.grid(True)
    plt.show()

    print(f"\nExploratory Visualization created (Correlation: {correlation:.2f}).")
else:
    print("\nVisualization Error: One or more feature/target column names are incorrect. Check df.head() output for correct names.")