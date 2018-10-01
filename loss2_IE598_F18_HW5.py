# coding: utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import seaborn as sns
# from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

# EDA: Wine Dataset Header & Description
print(df_wine.head())
print(df_wine.describe())

# EDA: Correlation Matrix
cm = np.corrcoef(df_wine[df_wine.columns].values.T)
sns.set()
hm = sns.heatmap(cm, cbar=True, annot=False, square=False, fmt='.2f', annot_kws={'size': 15},
                 yticklabels=df_wine.columns, xticklabels=df_wine.columns)
plt.suptitle("Feature Heat Map")
plt.show()

# Splitting the data into 80% training and 20% test subsets.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ## Logistic Classifier [untransformed]
# training set
lr = LogisticRegression(multi_class='ovr', solver='liblinear')
lr.fit(X_train, y_train)
lr_train_pred = lr.predict(X_train)
print("LR Accuracy Score (train): ", metrics.accuracy_score(y_train, lr_train_pred))
print(metrics.confusion_matrix(y_train, lr_train_pred))
# testing set
lr_pred = lr.predict(X_test)
print("LR Accuracy Score (test): ", metrics.accuracy_score(y_test, lr_pred))
print(metrics.confusion_matrix(y_test, lr_pred))


# ## SVM Classifier [untransformed]
# training set
svm = LinearSVC()
svm.fit(X_train, y_train)
svm_train_pred = svm.predict(X_train)
print("SVM Accuracy Score (train): ", metrics.accuracy_score(y_train, svm_train_pred))
print(metrics.confusion_matrix(y_train, svm_train_pred))
# testing set
svm_pred = svm.predict(X_test)
print("SVM Accuracy Score (test): ", metrics.accuracy_score(y_test, svm_pred))
print(metrics.confusion_matrix(y_test, svm_pred))


# standarize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# ## Principal Component Analysis in scikit-learn
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# LR Classifier using PCA
lr_pca = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
print("LR-PCA Accuracy Score (train): ", lr_pca.score(X_train_pca, y_train))
print("LR-PCA Accuracy Score (test): ", lr_pca.score(X_test_pca, y_test))

# SVM Classifier using PCA
svm_pca = LinearSVC()
svm_pca.fit(X_train_pca, y_train)
print("SVM-PCA Accuracy Score (train): ", svm_pca.score(X_train_pca, y_train))
print("SVM-PCA Accuracy Score (test): ", svm_pca.score(X_test_pca, y_test))



# ## Linear Discriminant Analysis in scikit-learn
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

# LR Classifier using LDA
lr_lda = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_lda.fit(X_train_lda, y_train)
print("LR-LDA Accuracy Score (train): ", lr_lda.score(X_train_lda, y_train))
print("LR-LDA Accuracy Score (test): ", lr_lda.score(X_test_lda, y_test))

# SVM Classifier using LDA
svm_lda = LinearSVC()
svm_lda.fit(X_train_lda, y_train)
print("SVM-LDA Accuracy Score (train): ", svm_lda.score(X_train_lda, y_train))
print("SVM-LDA Accuracy Score (test): ", svm_lda.score(X_test_lda, y_test))


# ## kPCA in scikit-learn
scikit_kpca = KernelPCA(n_components=4, kernel='rbf', gamma=1)
X_train_kpca = scikit_kpca.fit_transform(X_train_std)
X_test_kpca = scikit_kpca.transform(X_test_std)

# kPCA logistic classifier
lr_kpca = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_kpca.fit(X_train_kpca, y_train)
print("LR-kPCA Accuracy Score (train): ", lr_kpca.score(X_train_kpca, y_train))
print("LR-kPCA Accuracy Score (test): ", lr_kpca.score(X_test_kpca, y_test))

# kPCA SVM classifier
svm_kpca = LinearSVC()
svm_kpca.fit(X_train_kpca, y_train)
print("SVM-kPCA Accuracy Score (train): ", svm_kpca.score(X_train_kpca, y_train))
print("SVM-kPCA Accuracy Score (test): ", svm_kpca.score(X_test_kpca, y_test))


#######################################################################################################################
print()
print()
print("My name is Joseph Loss")
print("My NetID is: loss2")
print("I hereby certify that I have read the University policy on Academic Integrity"
      " and that I am not in violation.")