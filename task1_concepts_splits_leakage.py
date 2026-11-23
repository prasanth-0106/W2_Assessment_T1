"""Task 1 - Concepts, Splits and Leakage."""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from loan_data import make_loan_data

def main():
    df = make_loan_data()
    X = df.drop(columns=["default", "application_date"])
    y = df["default"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=.50, random_state=42, stratify=y_temp
    )
    print(f"Split sizes -> train: {len(X_train)}, validation: {len(X_val)}, test (held aside): {len(X_test)}")
    numeric = ["age","income","employment_years","credit_score","loan_amount","debt_to_income"]
    categorical = ["employment_type","education","purpose","grade"]
    
    Xtr = X_train.copy()
    Xv = X_val.copy()
    for c in numeric:
        med = Xtr[c].median()
        Xtr[c] = Xtr[c].fillna(med); Xv[c] = Xv[c].fillna(med)
    for c in categorical:
        mode = Xtr[c].mode().iloc[0]
        Xtr[c] = Xtr[c].fillna(mode); Xv[c] = Xv[c].fillna(mode)
    Xtr = __import__("pandas").get_dummies(Xtr, columns=categorical)
    Xv = __import__("pandas").get_dummies(Xv, columns=categorical).reindex(columns=Xtr.columns, fill_value=0)

    depths = [1,3,5,10,None]
    train_scores, val_scores = [], []
    print("\nDecision tree overfitting demonstration")
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(Xtr, y_train)
        tr = accuracy_score(y_train, model.predict(Xtr))
        va = accuracy_score(y_val, model.predict(Xv))
        train_scores.append(tr); val_scores.append(va)
        print(f"Depth {str(depth):>4}: train {tr:.3f} val {va:.3f}")

    plt.figure(figsize=(8,5))
    labels = [str(d) for d in depths]
    plt.plot(labels, train_scores, marker="o", label="Train accuracy")
    plt.plot(labels, val_scores, marker="o", label="Validation accuracy")
    plt.xlabel("Decision tree max_depth"); plt.ylabel("Accuracy")
    plt.title("Overfitting: Train vs Validation Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("overfitting_train_validation.png", dpi=150); plt.close()

   
    cv_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    )
    
    cv_scores = cross_val_score(cv_model, X_train[numeric], y_train, cv=5, scoring="accuracy")
    print(f"\nLogisticRegression 5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    
    leak_X = df[numeric].copy()
    leak_X = leak_X.fillna(leak_X.median())
    scaled_before = StandardScaler().fit_transform(leak_X)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    correct_train = scaler.fit_transform(leak_X.iloc[idx_train])
    correct_test = scaler.transform(leak_X.iloc[idx_test])
    print("\nData leakage demonstration")
    print(f"Scaling before split: scaler mean for income = {StandardScaler().fit(leak_X).mean_[1]:.2f}")
    print(f"Correct train-only scaling: scaler mean for income = {scaler.mean_[1]:.2f}")
    print(f"Shapes -> leaked full matrix {scaled_before.shape}, correct train {correct_train.shape}, correct test {correct_test.shape}")
    print("Leakage explanation (4 lines)")
    print("1. Scaling before splitting calculates statistics using both training and test rows.")
    print("2. This lets information from the held-out test data influence preprocessing.")
    print("3. Correct practice is to fit the scaler on training data only and transform validation/test data.")
    print("4. A Pipeline applies this rule within each cross-validation fold and prevents leakage.")

    print("\nProblem classification")
    print("Spam detection: Supervised (labelled spam/ham examples)")
    print("Customer segmentation: Unsupervised (discover groups without target labels)")
    print("House price prediction: Supervised (known sale prices)")
    print("Anomaly detection: Usually unsupervised (identify unusual patterns without labels)")
    print("\nML workflow (7 steps)")
    for i, step in enumerate([
        "Define the business problem and target.",
        "Collect and inspect the data.",
        "Clean data and handle missing values/outliers.",
        "Explore data and engineer useful features.",
        "Split into train, validation and test sets.",
        "Train models and tune using validation/cross-validation.",
        "Evaluate once on the test set, communicate results and monitor."
    ], 1): print(f"{i}. {step}")
    print("\nPlot saved: overfitting_train_validation.png")

if __name__ == "__main__": main()
