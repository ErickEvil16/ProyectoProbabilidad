import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, geom, norm, uniform, expon

st.set_page_config(page_title='Simulador de distribuciones', layout='wide')
st.title('Simulador de distribuciones de probabilidad')
st.write('Herramienta para simular, analizar y visualizar distribuciones discretas y continuas.')

DISTRIBUCIONES = ['Binomial', 'Poisson', 'Geométrica', 'Normal', 'Uniforme continua', 'Exponencial']

def validar_muestra(n):
    if n <= 0:
        st.error('El tamaño de muestra debe ser mayor que cero.')
        st.stop()

def tabla_resultados(datos, media_teo, var_teo, n):
    media_emp = np.mean(datos)
    var_emp = np.var(datos, ddof=0)
    desv_emp = np.std(datos, ddof=0)
    desv_teo = np.sqrt(var_teo)
    return pd.DataFrame({
        'Concepto': ['Media', 'Varianza', 'Desviación estándar', 'Tamaño de muestra'],
        'Valor teórico': [media_teo, var_teo, desv_teo, n],
        'Valor simulado': [media_emp, var_emp, desv_emp, n],
        'Diferencia absoluta': [abs(media_emp-media_teo), abs(var_emp-var_teo), abs(desv_emp-desv_teo), 0]
    })

def interpretar(distribucion, tabla):
    dm = tabla.loc[0, 'Diferencia absoluta']
    dv = tabla.loc[1, 'Diferencia absoluta']
    n = int(tabla.loc[3, 'Valor simulado'])
    texto = f'En la distribución {distribucion}, los valores simulados se comparan con los valores teóricos esperados. '
    texto += f'Con una muestra de {n} datos, la diferencia en la media fue de {dm:.4f} y la diferencia en la varianza fue de {dv:.4f}. '
    texto += 'Estas diferencias aparecen porque la simulación usa datos aleatorios, por lo que los resultados no son exactamente iguales a los valores teóricos. '
    if n >= 1000:
        texto += 'Al usar un tamaño de muestra grande, los estadísticos simulados tienden a acercarse más a los teóricos, lo cual se relaciona con la Ley de los Grandes Números.'
    else:
        texto += 'Si se aumenta el tamaño de muestra, normalmente se observará una mejor aproximación entre la simulación y la teoría.'
    return texto

distribucion = st.sidebar.selectbox('Distribución', DISTRIBUCIONES)
n_muestra = st.sidebar.number_input('Tamaño de muestra', min_value=1, value=1000, step=100)
validar_muestra(n_muestra)

np.random.seed(None)

if distribucion == 'Binomial':
    ensayos = st.sidebar.number_input('Número de ensayos n', min_value=1, value=10, step=1)
    p = st.sidebar.number_input('Probabilidad de éxito p', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    datos = np.random.binomial(ensayos, p, n_muestra)
    media_teo = ensayos * p
    var_teo = ensayos * p * (1 - p)
    x = np.arange(0, ensayos + 1)
    y = binom.pmf(x, ensayos, p)
    titulo = f'Binomial(n={ensayos}, p={p})'
    tipo = 'discreta'
elif distribucion == 'Poisson':
    lam = st.sidebar.number_input('Parámetro lambda', min_value=0.0001, value=4.0, step=0.1)
    datos = np.random.poisson(lam, n_muestra)
    media_teo = lam
    var_teo = lam
    x = np.arange(0, max(int(np.max(datos)), int(lam + 4*np.sqrt(lam))) + 1)
    y = poisson.pmf(x, lam)
    titulo = f'Poisson(lambda={lam})'
    tipo = 'discreta'
elif distribucion == 'Geométrica':
    p = st.sidebar.number_input('Probabilidad de éxito p', min_value=0.0001, max_value=1.0, value=0.3, step=0.01)
    datos = np.random.geometric(p, n_muestra)
    media_teo = 1 / p
    var_teo = (1 - p) / (p ** 2)
    x = np.arange(1, max(int(np.max(datos)), 15) + 1)
    y = geom.pmf(x, p)
    titulo = f'Geométrica(p={p})'
    tipo = 'discreta'
elif distribucion == 'Normal':
    mu = st.sidebar.number_input('Media mu', value=0.0, step=0.1)
    sigma = st.sidebar.number_input('Desviación estándar sigma', min_value=0.0001, value=1.0, step=0.1)
    datos = np.random.normal(mu, sigma, n_muestra)
    media_teo = mu
    var_teo = sigma ** 2
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    y = norm.pdf(x, mu, sigma)
    titulo = f'Normal(mu={mu}, sigma={sigma})'
    tipo = 'continua'
elif distribucion == 'Uniforme continua':
    a = st.sidebar.number_input('Límite inferior a', value=0.0, step=0.1)
    b = st.sidebar.number_input('Límite superior b', value=1.0, step=0.1)
    if b <= a:
        st.error('El límite superior b debe ser mayor que el límite inferior a.')
        st.stop()
    datos = np.random.uniform(a, b, n_muestra)
    media_teo = (a + b) / 2
    var_teo = ((b - a) ** 2) / 12
    x = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), 400)
    y = uniform.pdf(x, loc=a, scale=b-a)
    titulo = f'Uniforme continua(a={a}, b={b})'
    tipo = 'continua'
else:
    lam = st.sidebar.number_input('Parámetro lambda', min_value=0.0001, value=1.5, step=0.1)
    datos = np.random.exponential(1/lam, n_muestra)
    media_teo = 1 / lam
    var_teo = 1 / (lam ** 2)
    x = np.linspace(0, max(np.max(datos), 5/lam), 400)
    y = expon.pdf(x, scale=1/lam)
    titulo = f'Exponencial(lambda={lam})'
    tipo = 'continua'

col1, col2 = st.columns([1.25, 1])
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    if tipo == 'discreta':
        valores, conteos = np.unique(datos, return_counts=True)
        ax.bar(valores, conteos / n_muestra, alpha=0.6, label='Frecuencia simulada')
        ax.plot(x, y, 'o-', label='PMF teórica')
        ax.set_ylabel('Probabilidad')
    else:
        ax.hist(datos, bins=30, density=True, alpha=0.6, label='Histograma simulado')
        ax.plot(x, y, linewidth=2, label='PDF teórica')
        ax.set_ylabel('Densidad')
    ax.set_title(titulo)
    ax.set_xlabel('Valores de la variable aleatoria')
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

with col2:
    tabla = tabla_resultados(datos, media_teo, var_teo, n_muestra)
    st.subheader('Comparación numérica')
    st.dataframe(tabla, use_container_width=True)
    st.subheader('Interpretación')
    st.write(interpretar(distribucion, tabla))

st.subheader('Datos simulados')
st.dataframe(pd.DataFrame({'X': datos}).head(100), use_container_width=True)

csv = tabla.to_csv(index=False).encode('utf-8')
st.download_button('Descargar tabla comparativa CSV', csv, 'resultados_distribucion.csv', 'text/csv')
