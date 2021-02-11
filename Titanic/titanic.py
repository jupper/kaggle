import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path = 'train.csv'
path_test = 'test.csv'
data = pd.read_csv(path)
data_test = pd.read_csv(path_test)

data['alone'] = np.where(data['SibSp'] + data['Parch'] > 0, 0, 1)
data_test['alone'] = np.where(data_test['SibSp'] + data_test['Parch'] > 0, 0, 1)

y = data.Survived

features = ['Pclass', 'Sex', 'Age', 'alone']
X = data[features]
X_test = data_test[features]

s = (X.dtypes == 'object')
object_cols = list(s[s].index)
label_X_train = X.copy()
label_X_test = X_test.copy()

label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X[col])
    label_X_test[col] = label_encoder.fit_transform(X_test[col])

my_imputer = SimpleImputer()
imputed_X = pd.DataFrame(my_imputer.fit_transform(label_X_train))
imputed_X.columns = label_X_train.columns
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(label_X_test))


train_X, val_X, train_y, val_y = train_test_split(imputed_X, y, random_state=0)


best_i = 0
best_value = 0
for i in range(1, 100):
    model = RandomForestClassifier(n_estimators=i)
    model.fit(train_X, train_y)
    prediction = model.predict(val_X)
    tmp = model.score(val_X, val_y)
    if tmp > best_value:
        best_value = tmp
        best_i = i
    print(i)
    print(model.score(val_X, val_y))
print("Bester Lauf: " + str(best_value) + " mit n = " + str(best_i))
'''
model = RandomForestClassifier(n_estimators=34)
model.fit(train_X, train_y)
predictions = model.predict(imputed_X_test)
output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
'''