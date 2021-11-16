import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from pandas import DataFrame
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

classifier= KNeighborsClassifier(n_neighbors=3)
fig = plt.subplots()


df = pd.read_csv('airline_passenger_satisfaction.csv')
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.drop(['type_of_travel'], axis = 1, inplace = True)
df.drop(['customer_class'], axis = 1, inplace = True)
df.drop(['age'], axis = 1, inplace = True)
df.drop(['Gender'], axis = 1, inplace = True)

labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])

train, test=train_test_split(df,test_size=0.2)
max_k =train.shape[0]
k_list=[]
for i in range(3, max_k, 2):
    k_list.append(1)

#교차검증
cross_validation_scores = []
x_train = train[['flight_distance','inflight_wifi_service','departure_arrival_time_convenient','ease_of_online_booking','gate_location','food_and_drink','online_boarding','seat_comfort','inflight_entertainment','onboard_service','leg_room_service','baggage_handling','checkin_service','inflight_service','cleanliness','departure_delay_in_minutes','arrival_delay_in_minutes']]
y_train = train[['satisfaction']]

for k in k_list:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,x_train,y_train.value.ravel(), cv=10, scoring="accuracy")

    cross_validation_scores.append(scores.mean())

optimal_k = k_list[cross_validation_scores.index(max(cross_validation_scores))]

knn=KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train, y_train.values.ravel())

x_test = train[['flight_distance','inflight_wifi_service','departure_arrival_time_convenient','ease_of_online_booking','gate_location','food_and_drink','online_boarding','seat_comfort','inflight_entertainment','onboard_service','leg_room_service','baggage_handling','checkin_service','inflight_service','cleanliness','departure_delay_in_minutes','arrival_delay_in_minutes']]
y_test = train[['satisfaction']]

pred =knn.predict(x_test)

print("Accuracy : {}".format(accuracy_score(y_test.values.ravel(), pred)))

comparison = pd.DataFrame({
    "Prediction" : pred,

})
print(comparison)