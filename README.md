import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

activity_data = {
    'customer_name': ['Ana', 'Ben', 'Carla', 'Diego', 'Ella', 'Franz', 'Gina', 'Hector', 'Ivy', 'jomar', 'Kris', 'Lorna', 'Mike', 'Nina', 'Omar', 'Paula'],
    
    'browse_time': [5, 10 ,15, 20, 25, 30, 35,40, 8, 12, 18, 22, 28, 32, 38, 45],
    'cart_value_php': [150, 250, 350, 500, 600, 750, 900, 1000, 200, 300, 400, 550, 700, 800, 950, 1100],
    'purchased': [0, 0, 0, 1, 1, 1, 1, 1,0, 0, 1, 1, 1, 1, 1, 1]
    
    }


    
df = pd.DataFrame(activity_data)
df['status'] = df['purchased'].map({1: 'Buyer', 0: 'Non-Buyer'})
print(df[['customer_name','browse_time','cart_value_php','status']])

X = df[['browse_time', 'cart_value_php']]
y = df ['purchased']
names = df['customer_name']

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, names, test_size=0.25, random_state=42)

print("Training Group")
print(list(names_train))

print("\n Testing Group")
print(list(names_test))



model = DecisionTreeClassifier (max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Model Trained on", len(X_train), "Customers")



y_pred = model.predict(X_test)

results = pd.DataFrame({
    "customer_name": names_test.values,
    "actual": y_test.map({1: "Buyer", 0: "Non-Buyer"}).values,
    "predicted": pd.Series(y_pred).map({1: "Buyer", 0: "Non-Buyer"}).values
})

print(results)



acc = accuracy_score(y_test, y_pred)
print("Accuracy", round(acc*100,2), "%")
print(classification_report(y_test, y_pred, target_names=['Non-Buyer', "Buyer"]))
