#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Dosya konumuna ulaşıldı
data=pd.read_excel('/Users/onurkck/Desktop/PythonData/Data/GeneralLearn.xlsx')

print(data.head(3))

# Label Encoding işlemi
data['Region'] = data['Region'].map({'İstanbu': 1, 'Female': 0})

# X ve y değişkenlerini belirleme
X = data[['Product_Prise']]
y = data[['Sales_Quantity']]

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # %80 eğit  %20 tahmin et

# Lineer regresyon modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Eğitim ve test veri setleri üzerinde tahmin yapma
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Eğitim ve test veri setleri için doğruluk oranı hesaplama
accuracy_train = model.score(X_train, y_train)
accuracy_test = model.score(X_test, y_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print('Eğitim Seti Doğruluk Oranı:', accuracy_train)
print('Test Seti Doğruluk Oranı:', accuracy_test)
print('Eğitim Seti R² Değeri:', r2_train)
print('Test Seti R² Değeri:', r2_test)

#Neyi tahmin edeceksin
prediction_april_sales = model.predict([[20]])
print('Nisan satış miktarı tahmini:', prediction_april_sales)

# Görselleştirmeye lojistik regresyon eğrisini ekleyin
plt.scatter(data['Product_Prise'], data['Sales_Quantity'], label='Gerçek Satış')
plt.plot(data['Product_Prise'], model.predict(X), color='red', label='Tahmini Satış')
plt.xlabel('Product_Prise')
plt.ylabel('Sales_Quantity')
plt.title('Lineer Regresyon Modeli ile Tahmini Satış')
plt.legend()
plt.show()
