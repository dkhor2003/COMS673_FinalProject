import glob
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.metrics import accuracy_score
import argparse
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logistic','linear','perceptron'], required=True, help='Choose the model')
parser.add_argument('--test_only', action='store_true', help='Only perform testing using trained models')
args = parser.parse_args()

save_dir = 'model'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, f"{args.model}_model.pkl")

print('loading data...')
all_files = glob.glob("data/g1_traj/*.csv")
df_list = []
for fn in all_files:
    df = pd.read_csv(fn)
    df["time"] = np.arange(len(df)) * 0.02
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
X = data.drop(columns=["fallover"])
y = data["fallover"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not args.test_only:
    print(f'training model {args.model}...')
    if args.model == 'logistic':
        clf = LogisticRegression(max_iter=10_000)
    elif args.model == 'linear':
        clf = LinearRegression()
    else:
        clf = Perceptron()
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
else:
    clf = joblib.load(model_path)

print(f'testing model {args.model}...')
if args.model == 'linear':
    pred = (clf.predict(X_test) > 0.5).astype(int)
else:
    pred = clf.predict(X_test)
print("Accuracy on test data:", accuracy_score(y_test, pred))

if args.model == 'linear':
    pred = (clf.predict(X) > 0.5).astype(int)
else:
    pred = clf.predict(X)
# Compute accuracy on all data
print("Accuracy on all data:", accuracy_score(y, pred))

# Compute confusion matrix-based metrics
cm = confusion_matrix(y, pred)
tn, fp, fn, tp = cm.ravel()
tpr_val = tp / (tp + fn)
fpr_val = fp / (fp + tn)
print("TPR:", tpr_val)
print("FPR:", fpr_val)
