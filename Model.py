import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import accuracy_score, classification_report

Sample synthetic dataset

data = { 'GPA': np.random.uniform(2.0, 4.0, 1000), 'Attendance': np.random.uniform(50, 100, 1000), 'Test_Scores': np.random.uniform(40, 100, 1000), 'Socioeconomic_Status': np.random.choice(['Low', 'Middle', 'High'], 1000), 'Past_Enrollments': np.random.randint(0, 5, 1000), 'Enrollment_Status': np.random.choice([0, 1], 1000)  # 1 = Enrolled, 0 = Not Enrolled }

df = pd.DataFrame(data)

Convert categorical variables to numerical

df = pd.get_dummies(df, columns=['Socioeconomic_Status'], drop_first=True)

Define features and target

X = df.drop(columns=['Enrollment_Status']) y = df['Enrollment_Status']

Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train a Random Forest model

model = RandomForestClassifier(n_estimators=100, random_state=42) model.fit(X_train, y_train)

Predictions

y_pred = model.predict(X_test)

Evaluate model

accuracy = accuracy_score(y_test, y_pred) print("Accuracy:", accuracy) print("Classification Report:\n", classification_report(y_test, y_pred))

