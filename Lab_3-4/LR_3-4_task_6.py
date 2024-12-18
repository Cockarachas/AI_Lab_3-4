import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Завантаження даних з файлу
data = pd.read_csv('data_multivar_nb.txt')
data.columns = ['feature1', 'feature2', 'label']

# Розділення на ознаки та мітки класів
X = data[['feature1', 'feature2']]
y = data['label']

# Розділення даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Побудова моделі SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Побудова наївного байєсівського класифікатора
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Оцінка моделей
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("Accuracy SVM:", accuracy_score(y_test, y_pred_svm))

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Accuracy Naive Bayes:", accuracy_score(y_test, y_pred_nb))

# Візуалізація результатів
plt.figure(figsize=(12, 5))
sns.scatterplot(data=data, x='feature1', y='feature2', hue='label')
plt.title("Data Distribution by Feature1 and Feature2")
plt.show()
