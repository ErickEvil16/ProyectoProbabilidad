import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom, poisson, geom, norm, uniform, expon
from pathlib import Path

out = Path('imagenes')
out.mkdir(exist_ok=True)
res = []
np.random.seed(42)

def guardar(nombre, datos, media_teo, var_teo, fig):
    path = out / f'{nombre}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    res.append({
        'Distribución': nombre,
        'Media teórica': media_teo,
        'Media simulada': np.mean(datos),
        'Varianza teórica': var_teo,
        'Varianza simulada': np.var(datos, ddof=0),
        'Desv. estándar teórica': np.sqrt(var_teo),
        'Desv. estándar simulada': np.std(datos, ddof=0),
        'Tamaño de muestra': len(datos)
    })

n=1000
# Binomial
ensayos,p=10,0.5; datos=np.random.binomial(ensayos,p,n); x=np.arange(0,ensayos+1); y=binom.pmf(x,ensayos,p)
fig,ax=plt.subplots(figsize=(7,4)); v,c=np.unique(datos,return_counts=True); ax.bar(v,c/n,alpha=.6,label='Frecuencia simulada'); ax.plot(x,y,'o-',label='PMF teórica'); ax.set(title='Binomial(n=10, p=0.5)',xlabel='X',ylabel='Probabilidad'); ax.legend(); ax.grid(alpha=.25); guardar('Binomial',datos,ensayos*p,ensayos*p*(1-p),fig)
# Poisson
lam=4; datos=np.random.poisson(lam,n); x=np.arange(0,max(datos)+1); y=poisson.pmf(x,lam)
fig,ax=plt.subplots(figsize=(7,4)); v,c=np.unique(datos,return_counts=True); ax.bar(v,c/n,alpha=.6,label='Frecuencia simulada'); ax.plot(x,y,'o-',label='PMF teórica'); ax.set(title='Poisson(lambda=4)',xlabel='X',ylabel='Probabilidad'); ax.legend(); ax.grid(alpha=.25); guardar('Poisson',datos,lam,lam,fig)
# Geométrica
p=.3; datos=np.random.geometric(p,n); x=np.arange(1,max(datos)+1); y=geom.pmf(x,p)
fig,ax=plt.subplots(figsize=(7,4)); v,c=np.unique(datos,return_counts=True); ax.bar(v,c/n,alpha=.6,label='Frecuencia simulada'); ax.plot(x,y,'o-',label='PMF teórica'); ax.set(title='Geométrica(p=0.3)',xlabel='X',ylabel='Probabilidad'); ax.legend(); ax.grid(alpha=.25); guardar('Geometrica',datos,1/p,(1-p)/p**2,fig)
# Normal
mu,sigma=0,1; datos=np.random.normal(mu,sigma,n); x=np.linspace(-4,4,400); y=norm.pdf(x,mu,sigma)
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(datos,bins=30,density=True,alpha=.6,label='Histograma simulado'); ax.plot(x,y,label='PDF teórica'); ax.set(title='Normal(mu=0, sigma=1)',xlabel='X',ylabel='Densidad'); ax.legend(); ax.grid(alpha=.25); guardar('Normal',datos,mu,sigma**2,fig)
# Uniforme
A,B=0,10; datos=np.random.uniform(A,B,n); x=np.linspace(-1,11,400); y=uniform.pdf(x,loc=A,scale=B-A)
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(datos,bins=30,density=True,alpha=.6,label='Histograma simulado'); ax.plot(x,y,label='PDF teórica'); ax.set(title='Uniforme continua(a=0, b=10)',xlabel='X',ylabel='Densidad'); ax.legend(); ax.grid(alpha=.25); guardar('Uniforme',datos,(A+B)/2,(B-A)**2/12,fig)
# Exponencial
lam=1.5; datos=np.random.exponential(1/lam,n); x=np.linspace(0,max(datos),400); y=expon.pdf(x,scale=1/lam)
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(datos,bins=30,density=True,alpha=.6,label='Histograma simulado'); ax.plot(x,y,label='PDF teórica'); ax.set(title='Exponencial(lambda=1.5)',xlabel='X',ylabel='Densidad'); ax.legend(); ax.grid(alpha=.25); guardar('Exponencial',datos,1/lam,1/lam**2,fig)

pd.DataFrame(res).to_csv('resultados/tabla_comparativa.csv', index=False)
