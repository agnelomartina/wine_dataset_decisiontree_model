
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target
print(X,y)
# train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
model= DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy score(test accuracy)
print("Accuracy", accuracy_score(y_test, y_pred))

# Precision, Recall, F1-score (macro-averaged)
precision = precision_score(y_test, y_pred,average="macro")
recall = recall_score(y_test, y_pred,average="macro")
f1 = f1_score(y_test, y_pred,average="macro")

print("Precision", precision)
print("Recall", recall)
print("F1 Score", f1)

# Full classification report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

#cross validation
scores = cross_val_score(model, X, y, cv=5)
print("cross validation score",scores)

#dataframe for visualisation
data_wine=load_wine()
X1=pd.DataFrame(data_wine.data,columns=data_wine.feature_names)
Y1=pd.DataFrame(data_wine.target)
print(X1,Y1)
df=pd.concat([X1,Y1],axis=1)
#heatmap
""" df.corr()->Creates correlation matrix between all numeric columns
sns.heatmap()->Draws a visual correlation grid
annot=True->Shows numbers inside each cell
cmap='coolwarm'->Sets color range (blue → red)
fmt=".2f"->Formats numbers to 2 decimal places"""

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Wine Dataset")
plt.show()

#scatterplot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='alcohol', y='color_intensity')
plt.title("alcohol vs color intensity")
plt.show()