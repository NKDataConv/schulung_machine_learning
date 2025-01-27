import pandas as pd

df = pd.read_csv("daten/weatherAUS.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

print(df.head())
print(df.columns)

# Hoechste Temperatur
hoechste_temp = df["MaxTemp"].max()
print("Die höchste Temperatur war ", hoechste_temp)
mask = df["MaxTemp"] == hoechste_temp
print(df.loc[mask, :])
print(df.loc[mask, "Location"])

# Stadt mit höchstem Durchschnitt:
durchschnitt_temp = df.groupby("Location").agg({"MaxTemp": "mean"})
print(durchschnitt_temp)
durchschnitt_temp.sort_values(by="MaxTemp")
# alternativ:
hoechste_wert = durchschnitt_temp["MaxTemp"].max()
mask = durchschnitt_temp["MaxTemp"] == hoechste_wert
print(durchschnitt_temp.loc[mask, :])

# Niedrigste Schwankung:
df_agg_std = df.groupby("Location").agg({"MaxTemp": "std"})
df_agg_std = df_agg_std.sort_values(by="MaxTemp")
print(df_agg_std.iloc[0, :])


# Aufgabe 4: Kälteste Monat:
df["monat"] = df.index.month
df_agg_monat = df.groupby("monat").agg({"MinTemp": "mean"})
print(df_agg_monat)

# meiste Regen
print(df.columns)
df_agg_regen = df.groupby("monat").agg({"Rainfall": "sum"})
df_agg_regen = df_agg_regen.sort_values(by="Rainfall")
print(df_agg_regen)

# Erhöhung der Temperatur
df["jahr"] = df.index.year
df_agg = df.groupby("jahr").agg({"MaxTemp": "mean"})
print(df_agg)

# Anzahl Werte pro Jahr
df["jahr"].value_counts()