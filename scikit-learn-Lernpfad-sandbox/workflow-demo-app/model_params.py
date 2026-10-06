from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Alle verfügbaren Parameter anzeigen
print("LogReg Parameter:")
print(LogisticRegression().get_params().keys())

print("\nLinearSVC Parameter:")
print(LinearSVC().get_params().keys())

print("\nRandomForest Parameter:")
print(RandomForestClassifier().get_params().keys())

print("\nNaiveBayes Parameter:")
print(MultinomialNB().get_params().keys())