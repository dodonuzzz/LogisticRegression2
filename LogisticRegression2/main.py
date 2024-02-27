from sklearn import datasets,linear_model,metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

x = digits.data
y = digits.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 1)

reg = linear_model.LogisticRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

print("Lojistik Regresyon Doğruluğu : % ",metrics.accuracy_score(y_test,y_pred)*100)

