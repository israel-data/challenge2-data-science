import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga del archivo JSON (simulando extracción desde API)
ruta = "TelecomX_Data.json"
data = pd.read_json(ruta)

# Normalización del JSON (estructura anidada)
df = pd.json_normalize(data.to_dict(orient="records"),
    sep="_"
)
df.info()

df.dtypes

# Limpieza y estandarización
# Normalizar nombres de columnas
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

# Normalizar valores de texto
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.lower().str.strip()
    
# Manejo de inconsistencias y valores nulos

# Eliminar duplicados
df.drop_duplicates(inplace=True)

# Reemplazar valores vacíos por NaN
df.replace("", np.nan, inplace=True)

# Rellenar valores nulos numéricos con 0
df.fillna(0, inplace=True)

# Conversión de variables binarias

# Conversión de churn
df["churn"] = df["churn"].replace({"yes": 1, "no": 0})
df["churn"] = df["churn"].infer_objects(copy=False).astype(int)

# Variables tipo sí/no
binarias = [
    "phone_phoneservice",
    "internet_onlinesecurity",
    "internet_onlinebackup",
    "internet_deviceprotection",
    "internet_techsupport",
    "internet_streamingtv",
    "internet_streamingmovies"
]

for col in binarias:
    if col in df.columns:
        df[col] = df[col].replace({"yes": 1, "no": 0})
        df[col] = df[col].infer_objects(copy=False)
   
# Columna Opcional: Cuentas Diarias 
col_monthly = [c for c in df.columns if "monthly" in c and "charge" in c]

if "account_monthly_charges" in df.columns:
    df["cuenta_diaria"] = df["account_monthly_charges"] / 30

#Carga y Análisis (L – Load & Analysis)
# Análisis descriptivo

print(df.describe())

# Distribución de Evasión (Churn)
df["churn"].value_counts().plot(kind="bar")
plt.title("Distribución de Evasión de Clientes (Churn)")
plt.xlabel("Churn (0 = Activo, 1 = Baja)")
plt.ylabel("Número de Clientes")
plt.show()

# Evasión por Variables Categóricas
col_contract = [c for c in df.columns if "contract" in c][0]

pd.crosstab(df[col_contract], df["churn"]).plot(kind="bar")
plt.title("Evasión por Tipo de Contrato")
plt.xlabel("Tipo de Contrato")
plt.ylabel("Clientes")
plt.show()

# Método de pago
col_payment = [c for c in df.columns if "payment" in c][0]

pd.crosstab(df[col_payment], df["churn"]).plot(kind="bar")
plt.title("Evasión por Método de Pago")
plt.xlabel("Método de Pago")
plt.ylabel("Clientes")
plt.show()

# Antigüedad del cliente
col_tenure = [c for c in df.columns if "tenure" in c][0]

df.boxplot(column=col_tenure, by="churn")
plt.title("Antigüedad del Cliente vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Meses de Permanencia")
plt.show()

# Total gastado
# Identificar columna de total charges
col_total = [c for c in df.columns if "total" in c and "charge" in c][0]

# Convertir a numérico (maneja errores correctamente)
df[col_total] = pd.to_numeric(df[col_total], errors="coerce")

# Reemplazar NaN resultantes
df[col_total] = df[col_total].fillna(0)
df.boxplot(column=col_total, by="churn")
plt.title("Total Gastado vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Total Gastado")
plt.show()

# Extra: Análisis de Correlación
corr = df.select_dtypes(include=["int64", "float64"]).corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Matriz de Correlación")
plt.show()
