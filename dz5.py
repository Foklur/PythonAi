import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("internship_candidates_final_numeric.csv")
X = df[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]]
y = df["Accepted"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report ::\n", classification_report(y_test, y_pred))

english_levels = np.arange(1, 4)
test_scores = np.linspace(df.EntryTestScore.min(), df.EntryTestScore.max(), 100)

grid = pd.DataFrame([(el, ts) for el in english_levels for ts in test_scores], columns=["EnglishLevel", "EntryTestScore"])
grid["Experience"] = df["Experience"].mean()
grid["Grade"] = df["Grade"].mean()
grid["Age"] = df["Age"].mean()

grid["Probability"] = model.predict_proba(grid[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]])[:, 1]


plt.figure(figsize=(10, 6))
sns.lineplot(data=grid, x="EntryTestScore", y="Probability", hue="EnglishLevel", palette="Set1")
plt.title("Ймовірність прийняття в залежності від рівня англійської та балів за тест")
plt.xlabel("Бали за вступний тест")
plt.ylabel("Ймовірність прийняття")
plt.legend(title="Рівень англійської мови")
plt.grid(True)
plt.tight_layout()
plt.show()