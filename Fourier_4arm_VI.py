#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:17:37 2024

@author: alberto
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from uncertainties import unumpy,ufloat

# M22 -> ["36683","36683","36683","36683","36683","36683"]

# Leemos los datos de las fases de nuestra curva de luz, calculados con Fases_VI.py
datosV=pd.read_csv('/home/ces/YES/noseM22/OGLE-BLG-RRLYR-36683_V.fas',
                  header = 0, delimiter=("\t"))
datosI=pd.read_csv('/home/ces/YES/noseM22/OGLE-BLG-RRLYR-36683_I.fas',
                  header = 0, delimiter=("\t"))

# Asignamos cada columna a un valor en especifico
HJDV=datosV['HJD']
faseV=datosV['fase']
magV=datosV['mag']

HJDI=datosI['HJD']
faseI=datosI['fase']
magI=datosI['mag']
# Damos el periodo y la epoca de maximo o minimo(eclipsantes)
P=0.402369 #0.377279  #0.702785969 #0.702786
E0=6479.80381 #2458280.8720 #2458280.871955127

"""En la siguiente parte se calcula el ajuste de Fourier con 6 armonicos, todo 
a partir de los resultados de la parte anterior"""

# Definimos la serie de Fourier a usar
def m(x, A0, A1, A2, A3, A4, phi1, phi2, phi3, phi4):
    return A0+A1*np.cos((2*np.pi/P)*(x-E0)+phi1) \
        +A2*np.cos(2*(2*np.pi/P)*(x-E0)+phi2) \
        +A3*np.cos(3*(2*np.pi/P)*(x-E0)+phi3) \
        +A4*np.cos(4*(2*np.pi/P)*(x-E0)+phi4) \
        
# Calculamos el ajuste de Fourier a la curva de luz y guardamos resultados
resV, covV=curve_fit(m,HJDV,magV,bounds=((0,0,0,0,0,0,0,0,0),
                   (np.inf,1,1,1,1,2*np.pi,2*np.pi,2*np.pi,2*np.pi)))

resI, covI=curve_fit(m,HJDI,magI,bounds=((0,0,0,0,0,0,0,0,0),
                   (np.inf,1,1,1,1,2*np.pi,2*np.pi,2*np.pi,2*np.pi)))
#print(res)
"""res es el resultado del ajuste, es decir, el punto cero (A0), las amplitudes(A1,..)
y los armonicos(phi1...) usados. cov es la matriz de covarianza, su diagonal es la
varianza de cada uno de los parametros. En otras palabras la raiz cuadrada de los 
valores de la diagonal es el error de cada una de las amplitudes y/o armonicos, 
respectivamente"""

# Creamos un vector para graficar el ajuste de Fourier
x=np.linspace(-0.3,1.3,321)
# Calculamos el HJD y su magnitud correspondiente a cada punto del vector x
t=x*P+E0

Vaj=m(t,resV[0],resV[1],resV[2],resV[3],resV[4],resV[5],resV[6],resV[7],resV[8])

Iaj=m(t,resI[0],resI[1],resI[2],resI[3],resI[4],resI[5],resI[6],resI[7],resI[8])

# Guardamos el ajuste de la curva de luz en cada filtro
ajusteV=pd.DataFrame({'fase':x,'mag':Vaj})
ajusteV.apply(lambda col: col.map('{:+.3f}'.format) if col.name == 'fase'
             else col.map('{:.5f}'.format)).to_csv(
    '/home/ces/YES/noseM22/OGLE-BLG-RRLYR-36683_V_AFou.dat',
    sep='\t',
    index=False,
    header=True
)
                 
ajusteI=pd.DataFrame({'fase':x,'mag':Iaj})
ajusteI.apply(lambda col: col.map('{:+.3f}'.format) if col.name == 'fase'
             else col.map('{:.5f}'.format)).to_csv(
    '/home/ces/YES/noseM22/OGLE-BLG-RRLYR-36683_I_AFou.dat',
    sep='\t',
    index=False,
    header=True
)

"""Graficamos los puntos de la curva de luz,creamos un subplot 2x1 y graficamos
 por separado para cada filtro, posteriormente le damos formato a la grafica"""
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(6.4, 6.4)) #creamos la grafica 2x1

# Graficamos para el filtro V y cambiamos los parametros de graficado
ax1.scatter(faseV,magV,marker='.',c='black')
ax1.plot(x,Vaj,color='r')
ylim=ax1.get_ylim()
ax1.text(0.5,ylim[0],'V2', fontsize=14,color='k',va='top',ha='center')
ax1.set_ylim([ylim[0] - 0.1, ylim[1] + 0.1])
ax1.set_xlim([-0.35, 1.35])
ax1.invert_yaxis()
ax1.set_xticks([0,0.5,1])
ax1.minorticks_on()
ax1.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
ax1.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
ax1.set_ylabel(r"V", fontstyle="italic",fontsize=14)
ax1.set_xticklabels([])

# Graficamos para el filtro I y cambiamos los parametros de graficado

ax2.scatter(faseI,magI,marker='.',c='black')
ax2.plot(x,Iaj,color='r')
ylim=ax2.get_ylim()
ax2.set_ylim([ylim[0] - 0.1, ylim[1] + 0.1])
ax2.set_xlim([-0.35, 1.35])
ax2.invert_yaxis()
ax2.set_xticks([0,0.5,1])
ax2.minorticks_on()
ax2.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
ax2.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
ax2.set_ylabel(r"I", fontstyle="italic",fontsize=14)
ax2.set_xlabel(r"$\phi$",fontsize=14)

# Hacemos los ultimos detalles del formato y guardamos
plt.rcParams['font.family'] = 'serif'
plt.subplots_adjust(hspace=0,left=0, right=1, top=1, bottom=0) 
plt.tight_layout(pad=0)
#plt.savefig('/home/alberto/Escritorio/Modular_Mariano/Ejemplos/V2_AFou.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

"""Vamos a guardar los valores de A y phi del resultado del ajuste de Fourier"""

error=np.sqrt(np.diag(covV)) # Error de a partir de la diagonal de la covarianza
coef=unumpy.uarray(resV,error) # Resultados con sus errores correspondientes

# Guardamos los armonicos de Fourier
arm_fou=pd.DataFrame([coef],columns=['A0','A1','A2','A3','A4', 'phi1', 
                 'phi2', 'phi3','phi4'])

"""Las siguientes iteraciones son para calcular los coeficientes de fourier: 
Rij=Ai/Aj y Phiij=j*phi[i-1]-i*phi[j-1] a partir de los valores de A y phi.
Para ello, separamos las amplitudes y los armonicos"""

n=int((len(resV)-1)/2+1) # Calculamos el numero total de las amplitudes
l=len(resV) # numero total de entradas en el resultado

# Guardamos por separado las A y los phi
A=coef[0:n] 
phi=coef[n:l]
#print(A)
#print(phi)
a=ufloat(-99.99,99.99)
A=np.append(A,[a,a])
phi=np.append(phi,[a,a])


"""A0 le podemos llamar intensity weigthed mean, con notacion <V> o <I>, 
segun el filtro"""
V=A[0] # Guardamos A0
I=ufloat(resI[0],np.sqrt(covI[0,0]))
# Realizamos las iteraciones

for i in [2,3,4]:
    for j in [1,2,3]:
        nombre_variable = f"R{i}{j}"  # Crear el nombre de la variable
        valor = A[i]/A[j]  # Calculamos el valor
        globals()[nombre_variable] = valor  # Guardamos el valor como variable global

# Definimos una funcion para que los valores de Phi queden en el rango deseado
def ajustar_valor_al_rango(valor, rango_min, rango_max):
    while valor < rango_min:
        valor += 2 *np.pi
    while valor > rango_max:
        valor -= 2 * np.pi
    return valor

for i in [2,3,4,5,6]:
    for j in [1,2,3]:
        nombre_variable = f"Phi{i}{j}" # Crear el nombre de la variable
        valor=j*phi[i-1]-i*phi[j-1]  # Calculamos el valor
        valor=ajustar_valor_al_rango(valor,0,2*np.pi) # El valor siempre sera ente 0 y 2pi
        globals()[nombre_variable] = valor # Guardamos el valor como variable global
       
# Guardamos los coeficientes de Fourier 
CF=[R21,R31,R41,R32,R42,R43,Phi21,Phi31,Phi41,Phi32,Phi42,Phi43]
coef_four=pd.DataFrame([CF],columns=['R21','R31','R41','R32','R42','R43',
                                     'Phi21','Phi31','Phi41','Phi32','Phi42',
                                     'Phi43'])
# Lo siguiente no se si se quedara en la version final
sal=['V2',V,I,A[1],A[2],A[3],A[4],A[5],A[6],Phi21,Phi31,Phi41,Phi51,Phi61]
salida=pd.DataFrame([sal],columns=['ID','V','I','A1','A2','A3','A4','A5','A6',
                                     'Phi21','Phi31','Phi41','Phi51','Phi61'])


"""Guardamos los dataframe en archivos .dat qeu seran usados Dm_paraemter.py y 
los Par_fis_*.py"""
coef_four.to_csv('/home/ces/YES/noseM22/36683-Coef_Fou.dat'
                 , index=False, sep='\t', header=True)


salida.to_csv('/home/ces/YES/noseM22/36683-Salida.dat'
                 , index=False, sep='\t', header=True)