import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import batman
def train_random_forest(csv_file_path):
    """
    Loads the dataset from CSV, trains a Random Forest model, 
    and prints the classification report.
    """
    # 1. Load the CSV dataset
    df = pd.read_csv(csv_file_path)
    X = df.iloc[:, :-1].values  # All rows, excluding the last column (label)
    y = df['label'].values

    # 2. Extract simple features: mean and standard deviation
    #    You can replace this with more complex feature extraction if needed
    features = np.column_stack([X.mean(axis=1), X.std(axis=1)])

    # 3. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )

    # 4. Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Test and print the classification report
    y_pred = clf.predict(X_test)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_random_forest("synthetic_light_curves_with_batman.csv")
