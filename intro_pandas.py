import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("daten/BTC_daily.csv")
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
print(df.head())

# NaN Werte löschen
df = df.dropna()
print(df.head())

# Auf einzelne Spalten zugreifen
print(df["Open"])
print(df[["Open", "High"]])

# Auf einzelne Zeilen zugreifen
print(df.loc["2014-09-17"])
print(df.loc["2014-09-17", "Open"])
print(df.loc["2014-09-17", ["Open", "High"]])

# mit Indizies auf Elemente zugreifen
print(df.iloc[0, 0])
print(df.iloc[1, :])
print(df.iloc[1, :2])

# Der höchste Wert des Bitcoin
print(df.columns)
hoechster_Wert = df["High"].max()
print("Der höchste Wert des Bitcoin war ", hoechster_Wert)

# Die niedrigsten Werte des Bitcoin
niedrigster_Wert = df["Low"].min()
print("Der niedrigste Wert des Bitcoin war ", niedrigster_Wert)

mask = df["Low"] == niedrigster_Wert
print(df.loc[mask, :])

# An welchem Tag wurde am meisten gehandelt
max_volumen = df["Volume"].max()
mask = df["Volume"] == max_volumen
print(df.loc[mask, :])

# Der höchte Anstieg an einem Tag
print(df.columns)
df["anstieg"] = df["Close"] - df["Open"]
hoechster_anstieg = df["anstieg"].max()
print("Der höchste Anstieg an einem Tag war", hoechster_anstieg)

# der Durchschnitt aller Open Kurse
durchschnitt_open = df["Open"].mean()
print("Der Durchschnitt aller open Kurse war ", durchschnitt_open)

# Wenn am Anfang 1000€ investiert wurden, wie viel wären sie heute wert?
erster_tag = df.index.min()
erster_kurs = df.loc[erster_tag, "Open"]
anzahl_bitcoin = 1000 / erster_kurs
print("Wir hätten ", anzahl_bitcoin, " Bitcoin gekauft")

letzter_tag = df.index.max()
letzter_kurs = df.loc[letzter_tag, "Close"]
wert_der_bicoins = anzahl_bitcoin * letzter_kurs
print("Die 1000€ wären dann ", wert_der_bicoins, " wert")

# Wenn man jeden Tag 10€ investiert hätte, wie viel wären sie heute wert?
df["invest"] = 10
df["anzahl_bitcoins_gekauft"] = df["invest"] / df["Open"]
bitcoin_aufsummiert = df["anzahl_bitcoins_gekauft"].sum()
print("Insgesamt ", bitcoin_aufsummiert, "gekauft")

letzter_tag = df.index.max()
letzter_kurs = df.loc[letzter_tag, "Close"]
wert_der_bicoins = bitcoin_aufsummiert * letzter_kurs
print("Bei 10€ täglich:", wert_der_bicoins)

# Wieviel hätten wir ausgegeben?
ausgegeben = df.shape[0] * 10
ausgegeben = len(df) * 10
# alternativ:
ausgegeben = df["invest"].sum()
print("Insgesamt ausgegeben: ", ausgegeben)

# Gruppierungen
df["jahr"] = df.index.year
print(df)

df_agg = df.groupby("jahr").agg({"Low": "min"})
print(df_agg)

df["monat"] = df.index.month
df_agg_monat = df.groupby("monat").agg({"Volume": "mean"})
print(df_agg_monat)

df_agg_tag = df.groupby(df.index).agg({"Volume": "mean"})
print(df_agg_tag)

df_agg_monat_jahr = df.groupby(["jahr", "monat"]).agg({"Volume": "mean"})
print(df_agg_monat_jahr)

df = df.drop(columns=["jahr", "monat"])

# alternativ:
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

