import pandas as pd

df = pd.read_csv("daten/weatherAUS.csv")

df = df.dropna()

df = df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm", "Date", "Location"])
print(df.head())

print(df["RainToday"])
print(df["RainTomorrow"])

# rain_dict = {"No": 0, "Yes": 1}
# df["RainToday"] = df["RainToday"].map(rain_dict)
# df["RainTomorrow"] = df["RainTomorrow"].map(rain_dict)

# alternative:
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df["RainToday"])
df["RainToday"] = le.transform(df["RainToday"])
df["RainTomorrow"] = le.transform(df["RainTomorrow"])

risk_mm = df["RISK_MM"]
y = df["RainTomorrow"]
df = df.drop(columns=["RISK_MM", "RainTomorrow"])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df, y,
                                                    test_size=0.4,
                                                    random_state=42,
                                                    stratify=y)

x_test, x_vali, y_test, y_vali = train_test_split(x_test, y_test,
                                                    test_size=0.75,
                                                    random_state=42,
                                                    stratify=y_test)

print(x_train)
print(y_train)
