#!/usr/bin/env python
# coding: utf-8


# # ¿Cuál es la mejor tarifa?
# 
# Trabajas como analista para el operador de telecomunicaciones Megaline. La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate. El departamento comercial quiere saber cuál de las tarifas genera más ingresos para poder ajustar el presupuesto de publicidad.
# 
# Vas a realizar un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña. Tendrás los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018. Tu trabajo es analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos.

# 

# ## Inicialización

# In[3]:


# Cargar todas las librerías
import pandas as pd
import numpy as np
import math as mt
import seaborn as sns
from math import ceil
from math import factorial
from scipy import stats as st
from matplotlib import pyplot as plt


# ## Cargar datos

# In[4]:


# Carga los archivos de datos en diferentes DataFrames
df_megaline_calls = pd.read_csv("/datasets/megaline_calls.csv")
df_megaline_internet = pd.read_csv("/datasets/megaline_internet.csv")
df_megaline_messages = pd.read_csv("/datasets/megaline_messages.csv")
df_megaline_plans = pd.read_csv("/datasets/megaline_plans.csv")
df_megaline_users = pd.read_csv("/datasets/megaline_users.csv")



# ## Preparar los datos


# ## Tarifas

# In[5]:


# Imprime la información general/resumida sobre el DataFrame de las tarifas
df_megaline_plans.info()


# In[6]:


# Imprime una muestra de los datos para las tarifas
df_megaline_plans


# Lo primero que podemos observar, es que cuenta con solo dos filas, las cuales nos describen cada plan, en la primera fila podemos observar la información del plan de prepago Surf y en la siguiente el plan de prepago Ultimate. 

# ## Corregir datos

# 


# Como podemos observar en la tabla, nos encontramos con la columna "mb_per_month_included" que contiene el la cantidad de megabytes que Megaline ofrece como base para sus planes de prepago, pero debido a que en la descripción de los planes estos se encuentran como gigabytes, los datos serán modificados para que estos muestren gigabytes en lugar de megabytes.



# ## Usuarios/as

# In[7]:


# Imprime la información general/resumida sobre el DataFrame de usuarios
df_megaline_users.info()


# In[8]:


# Imprime una muestra de datos para usuarios
df_megaline_users.head(10)



# Por lo que se puede observar en el DataFame la mayoria de los datos ausentes estna en la columna "churn_date" la cual nos muestra la fecha en la que el usuario dejó de usar el servicio. Algo importante a considerad si el valor es ausente existe la posibilidad de que la tarifa se estaba usando cuando fue extraída esta base de datos.

# ### Corregir los datos


# In[9]:


df_megaline_users["churn_date"].fillna("acive", inplace=True)
print(df_megaline_users.head())


# In[47]:


df_megaline_users["reg_date"] = pd.to_datetime(df_megaline_users["reg_date"], format="%Y-%m-%d")
print(df_megaline_users["reg_date"].dtype)



# Cambie las filas con valores asentes por "active" para tener en cuenta que el plan de prepago de los usuarios esta activo, no elimine los datos debido a que estos representan una parte importante de todos los datos. Además se realizo un cambio en la columna "reg_date" para que esta sea un datatime.

# ### Enriquecer los datos



# In[15]:


df_megaline_users.duplicated().sum()


# Comprobé la existencia de duplicados, que en este caso son inexistentes. 

# ## Llamadas

# In[20]:


# Imprime la información general/resumida sobre el DataFrame de las llamadas
df_megaline_calls.info()


# In[21]:


# Imprime una muestra de datos para las llamadas
df_megaline_calls.head(10)



# Se puede observar qu el DataFrame cuenta con 137,735 filas que nos muestra el ID de las llamadas realizadas por cada usuarios, la duración de cada una de las llamadas y la fecha en la que se relizo cada llamada. Por lo que se puede observar no se encuentran datos ausentes.

# ### Corregir los datos

# In[18]:


df_megaline_calls["duration"] = np.ceil(df_megaline_calls["duration"])
print(df_megaline_calls["duration"])


# Cambie la columna "duration" de para redondear los minutos y segundos a solo minutos, ya que Megaline redondea los segundos a minutos.

# ### Enriquecer los datos


# In[48]:


df_megaline_calls["call_date"] = pd.to_datetime(df_megaline_calls["call_date"], format="%Y-%m-%d")
print(df_megaline_calls["call_date"].dtype)



# Realize un cambio en la columna "call_date" para transformar los datos a datetime.

# ## Mensajes

# In[25]:


# Imprime la información general/resumida sobre el DataFrame de los mensajes
df_megaline_messages.info()


# In[26]:


# Imprime una muestra de datos para los mensajes
df_megaline_messages.head(10)



# Se puede observar que el DataFrame cuenta con 3 columnas que nos muestran el ID de cada mensaje, así como el ID de cada uno de los usuarios y la fecha en la que se envió el mensaje. Cuenta con 76,051 filas. 

# ### Corregir los datos


# ### Enriquecer los datos

# In[49]:


df_megaline_messages["message_date"] = pd.to_datetime(df_megaline_messages["message_date"], format="%Y-%m-%d")
print(df_megaline_messages["message_date"].dtype)


# Realice un cambio en la columna "message_date" para trasformar los datos a datatime.

# ## Internet

# In[30]:


# Imprime la información general/resumida sobre el DataFrame de internet
df_megaline_internet.info()


# In[31]:


# Imprime una muestra de datos para el tráfico de internet
df_megaline_internet.head(15)



# Como se puede observar el DataFrame no cuenta con valores ausentes, en su contenido podemos encontrar la fecha de la sesión web y cuantos megabytes fueron consumidos por cada ocasión que fue utilizado, además de generar un ID para cada una de las sesiones.

# ### Corregir los datos

# In[34]:


df_megaline_internet["mb_used"] = (df_megaline_internet["mb_used"] / 1024).apply(np.ceil)
df_megaline_internet.rename(columns={"mb_used": "gb_used"}, inplace=True)
print(df_megaline_internet.head())


# Se realizó la conversión de megabytes a gigabytes, además de redondear los gigabytes, al igual se hizo un cambio en el nombre de la columna ya que ahora su contenido cambio a gigabytes, esto con la finalidad de evitar confusión en cuanto al contendido de los datos de la columna. 

# ### Enriquecer los datos

# In[50]:


df_megaline_internet["session_date"] = pd.to_datetime(df_megaline_internet["session_date"], format="%Y-%m-%d")
print(df_megaline_internet["session_date"].dtype)


# Realice una modificación en la columna "session_date" para que esta fuera de tipo datatime64[ns].


# ## Estudiar las condiciones de las tarifas


# In[36]:


# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras
df_megaline_plans


# ## Agregar datos por usuario
# 

# In[51]:


# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado.
df_megaline_calls["year_month"] = df_megaline_calls["call_date"].dt.to_period("M")
calls_per_month = df_megaline_calls.groupby(["user_id", "year_month"]).size().reset_index(name="num_calls")
print(calls_per_month)


# 

# In[52]:


# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado.
df_megaline_calls["year_month"] = df_megaline_calls["call_date"].dt.to_period("M")
minutes_per_month = df_megaline_calls.groupby(["user_id", "year_month"])["duration"].sum().reset_index(name="total_minutes")
print(minutes_per_month)


# In[53]:


# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
df_megaline_messages["year_month"] = df_megaline_messages["message_date"].dt.to_period("M")
messages_per_month = df_megaline_messages.groupby(["user_id", "year_month"]).size().reset_index(name= "num_messages")
print(messages_per_month)


# In[54]:


# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
df_megaline_internet["year_month"] = df_megaline_internet["session_date"].dt.to_period("M")
mb_per_month = df_megaline_internet.groupby(["user_id", "year_month"])["gb_used"].sum().reset_index(name="total_gb")
print(mb_per_month)


# In[55]:


# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month
merged_df = pd.merge(calls_per_month, messages_per_month, on=["user_id","year_month"])
merged_df = pd.merge(merged_df, minutes_per_month, on=["user_id","year_month"])
merged_df = pd.merge(merged_df, mb_per_month, on=["user_id","year_month"])
print(merged_df.head(10))


# Uní en una sola tabla los datos totales de llamadas, el total de minutos utilizados, el total de mensajes enviados y el total de megabytes utilizados por cada usuario y por cada mes.  

# In[56]:


# Añade la información de la tarifa
merged_df_1 = pd.merge(df_megaline_users, merged_df, on=["user_id"])
plan_cost = {'surf': 20, 'ultimate': 70}
merged_df_1["plan_cost"] = merged_df_1['plan'].map(plan_cost)
print(merged_df_1.head())


# Añadí el costo base que tiene cada plan, como corresponde por cada usuario y dependiendo del plan que el usuario a contratado.


# In[57]:


def additional_cost_mi(row):
    plan = row["plan"]
    minutes = row["total_minutes"]
    
    if plan == "surf":
        if minutes <= 500:
            return 0
        else:
            return (minutes - 500) * 0.03
    elif plan == "ultimate":
        if minutes <= 3000:
            return 0
        else:
            return (minutes - 3000) * 0.01


merged_df_1["income_per_minute"] = merged_df_1.apply(additional_cost_mi, axis=1)
print(merged_df_1.head())


# In[58]:


def additional_cost_me(row):
    plan = row["plan"]
    minutes = row["num_messages"]
    
    if plan == "surf":
        if minutes <= 50:
            return 0
        else:
            return (minutes - 50) * 0.03
    elif plan == "ultimate":
        if minutes <= 1000:
            return 0
        else:
            return (minutes - 1000) * 0.01


merged_df_1["income_per_message"] = merged_df_1.apply(additional_cost_me, axis=1)
print(merged_df_1.head())


# In[61]:


def additional_cost_in(row):
    plan = row["plan"]
    internet_gb = row["total_gb"]
    
    
    if plan == "surf":
        if internet_gb <= 15:  
            return 0
        else:
            return (internet_gb - 15) * 10
    elif plan == "ultimate":
        if internet_gb <= 30:  
            return 0
        else:
            return (internet_gb - 30) * 7  
    

merged_df_1["income_per_internet"] = merged_df_1.apply(additional_cost_in, axis=1)
print(merged_df_1.head())



# In[63]:


# Calcula el ingreso mensual para cada usuario
merged_df_1["total_income"] = merged_df_1[["plan_cost", "income_per_minute", "income_per_message", "income_per_internet"]].sum(axis=1)
print(merged_df_1.head())


# Realice las operaciones correspondientes para conocer el ingresos que representa el excedente de minutos, mensajes y gigabytes utilizado por cada cliente.

# ## Estudia el comportamiento de usuario


# ### Llamadas

# In[79]:


# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.
filtered_df = merged_df_1[merged_df_1["plan"].isin(["surf", "ultimate"])]

call_duration = filtered_df.groupby(["plan", "year_month"])["num_calls"].mean().reset_index()


fig, ax = plt.subplots(figsize=(15, 6))


for plan in ["surf", "ultimate"]:
    plan_data = call_duration[call_duration["plan"] == plan]
    ax.bar(plan_data["year_month"].astype(str), plan_data["num_calls"], alpha=0.5, label=plan)


ax.set_xlabel("Mes")
ax.set_ylabel("Número de llamadas")
ax.set_title("Duración promedio de llamadas por plan y mes")
ax.legend(title="Plan")

plt.show()


# La grafica compara el promedio de minutos utilizados por mes del plan Surf y del plan Ultimate.

# In[80]:


# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.

surf_data = merged_df_1[merged_df_1["plan"] == "surf"]
ultimate_data = merged_df_1[merged_df_1["plan"] == "ultimate"]

surf_min = surf_data.groupby(["user_id", "year_month"])["total_minutes"].sum().reset_index()
ultimate_min = ultimate_data.groupby(["user_id", "year_month"])["total_minutes"].sum().reset_index()

fig, ax = plt.subplots(figsize=(15, 6))

ax.hist(surf_min["total_minutes"], bins=20, alpha=0.5, label="Surf", color="blue")
ax.hist(ultimate_min["total_minutes"], bins=20, alpha=0.5, label="Ultimate", color="green")

ax.set_xlabel("Minutos totales")
ax.set_ylabel("Usuarios")
ax.set_title("Distribución de minutos totales por plan")
ax.legend(title="Plan")

plt.show()



# In[81]:


# Calcula la media y la varianza de la duración mensual de llamadas.
mean_calls = merged_df_1["total_minutes"].mean()
var_calls = np.var(merged_df_1["total_minutes"])
print(f"Media: {mean_calls}")
print(f"Varianza: {var_calls}")


# In[82]:


# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
monthly_call_duration = merged_df_1.groupby("year_month")["total_minutes"].sum()
sns.boxplot(monthly_call_duration)



# Los usuarios del plan Surf utilizan más sus minutos que los usuarios del plan Ultimate, el número de llamdas realizadas por los uriarios del plan Surf son muy similares que a los uasuarios del plan Ultimate excepto por tres meses (enero, febrero y marzo) ya que tiene una diferencia de llamadas entre ambos. En promedio se utilizan 445 minutos por usuario.  

# ### Mensajes

# In[83]:


# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
surf_messages = surf_data.groupby(["user_id", "year_month"])["num_messages"].sum().reset_index()
ultimate_messages = ultimate_data.groupby(["user_id", "year_month"])["num_messages"].sum().reset_index()

fig, ax = plt.subplots()

ax.hist(surf_messages["num_messages"], bins=15, alpha=0.5, label="Surf", color="blue")
ax.hist(ultimate_messages["num_messages"], bins=15, alpha=0.5, label="Ultimate", color="green")

ax.set_xlabel("Número de mensajes")
ax.set_ylabel("Número de usuarios")
ax.set_title("Número de mensajes enviados mensuales por plan")
ax.legend(title="Plan")

plt.show()


# In[84]:


mean_messages = merged_df_1["num_messages"].mean()
var_messages = np.var(merged_df_1["num_messages"])
print(f"Media: {mean_messages}")
print(f"Varianza: {var_messages}")


# In[85]:


monthly_messages_duration = merged_df_1.groupby("year_month")["num_messages"].sum()
sns.boxplot(monthly_messages_duration)


# Por lo que se puede obsevar en el frafico los usuarion que más utilizan SMS son los que han contratado el plan Surf, en comparacion a los que han optado por el plan Ultimate, en promedio se envian 42 SMS por usuario.

# ### Internet

# In[88]:


# Compara la cantidad de tráfico de Internet consumido por usuarios por plan
surf_internet = surf_data.groupby(["user_id", "year_month"])["total_gb"].sum().reset_index()
ultimate_internet = ultimate_data.groupby(["user_id", "year_month"])["total_gb"].sum().reset_index()

fig, ax = plt.subplots()

ax.hist(surf_internet["total_gb"], bins=15, alpha=0.5, label="Surf", color="blue")
ax.hist(ultimate_internet["total_gb"], bins=15, alpha=0.5, label="Ultimate", color="green")

ax.set_xlabel("Número de megabytes")
ax.set_ylabel("Número de usuarios")
ax.set_title("Tráfico de internet consumido al mes por plan")
ax.legend(title="Plan")

plt.show()


# In[90]:


mean_internet = merged_df_1["total_gb"].mean()
var_internet = np.var(merged_df_1["total_gb"])
print(f"Media: {mean_internet}")
print(f"Varianza: {var_internet}")


# In[92]:


monthly_internet_duration = merged_df_1.groupby("year_month")["total_gb"].sum()
sns.boxplot(monthly_internet_duration)



# Como podemos observar en la grafica los usuarios del pla Surf son los que han utilizado más gigabytes, en comparación con los usuarios que utilizan el plan Ultimate que utilizan menos gigabytes. se puede decir qu en promedio se utilizan 40 mb.

# ## Ingreso


# In[93]:


income_surf = surf_data.groupby("user_id")["total_income"].sum()
income_ultimate = ultimate_data.groupby("user_id")["total_income"].sum()

fig, ax = plt.subplots()

ax.hist(income_surf, alpha=0.5, label="Surf", color="blue")
ax.hist(income_ultimate, alpha=0.5, label="Ultimate", color="green")

ax.set_xlabel("Ingresos")
ax.set_ylabel("Número de usuarios")
ax.set_title("Ingreso total por plan")
ax.legend(title="Plan")

plt.show()



# En esta grafica nos podemos percatar que los ingresos por el plan de prepago Surf son considerablemente mayores a los del plan de prepago Ultimate, para esta grafica se tomó en cuenta tanto el ingreso base que es la cantidad cobrada inicialmente por cada uno de los planes y los ingresos extras que cada uno obtuvo por os minutos, mensajes y megabytes utilizados por cada usuario.


# ## Prueba las hipótesis estadísticas

# Hipótesis: son diferentes los ingresos promedio procedentes de los usuarios de los planes de llamada Ultimate y Surf.


# In[95]:


total_income_calls_s = surf_data.groupby("user_id")["income_per_minute"].sum()
total_income_calls_u = ultimate_data.groupby("user_id")["income_per_minute"].sum()


# In[96]:


# Prueba las hipótesis

alpha = 0.05

results = st.ttest_ind(total_income_calls_s, total_income_calls_u, equal_var= False) 

print('valor p: ', results.pvalue) 

if results.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# 


# Hipótesis: el ingreso promedio de los usuarios del área NY-NJ es diferente al de los usuarios de otras regiones.]


# In[97]:


# Prueba las hipótesis
users_ny = merged_df_1[merged_df_1["city"] == "New York-Newark-Jersey City, NY-NJ-PA MSA"]  
other_users = merged_df_1[merged_df_1["city"] != "New York-Newark-Jersey City, NY-NJ-PA MSA"]
income_ny = users_ny.groupby("user_id")["total_income"].sum()
income_other = other_users.groupby("user_id")["total_income"].sum()
alpha = 0.05

results_1 = st.ttest_ind(income_ny, income_other, equal_var= False)

print('valor p: ', results_1.pvalue) 

if results_1.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# 

# ## Conclusión general
# 

# Tomando en cuenta que el proposito de realizar el analisis de datos para el operador de telecomunicaciones Megaline, que tiene como finalidad analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos, de acuerdo a lo observado en el analisis realizado se puede concluir que los usuarios del plan de prepago utilizan más tanto los minutos, mensajes y megabytes, esto en comparacion a los usuarios del plan Ultimate, lo que se traduce que aunque sea mayor el pago por cada plan Ultimate contratado, en comparación con el plan Surf que se adquiere por 20$ son los usuarios que más exceden el limite que obtienen por su plan lo que se traduce en mayores ingresos. 

# 

