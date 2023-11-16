import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Mileage': [69000, 35000, 57000, 22500, 46000],
    'Year': [2010, 2016, 2012, 2018, 2015],
    'Brand': ['Toyota', 'Honda', 'Ford', 'Honda', 'Ford'],
    'Price': [6000, 12000, 8000, 15000, 10000]
}

df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['Brand'])
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

if len(y_test) >= 2:
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2}")
else:
    print("Not enough samples to calculate R-squared.")
