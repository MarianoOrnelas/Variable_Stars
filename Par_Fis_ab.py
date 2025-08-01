#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:02:31 2024

@author: alberto
"""

import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy
from uncertainties.unumpy import uarray, sqrt, log10

# Damos el periodo y la epoca de maximo o minimo(eclipsantes)
P=0.702785969 #0.702786
E0=2458280.8720 #2458280.871955127
EBV=0.02

#Leemos los datos y le damos el formato adecuado
coef_Fou = pd.read_csv('/home/alberto/Escritorio/Modular_Mariano/Ejemplos/Salida.dat', delimiter=("\t"))

def to_ufloat(text):
    return ufloat(*map(float, text.split('+/-')))

for column in coef_Fou.columns[1:]:
    coef_Fou[column] = coef_Fou[column].apply(to_ufloat)
    #print(column)

# Guardamos la variables con los nombres de cada columna para mejor manipulación
for column_name in coef_Fou.columns[1:]:
    globals()[column_name] = coef_Fou[column_name].values

"""Definimos la ecuacion para que los Phi queden en el rango deseado, cada Phiij
tiene su propio dominio y por eso es un valor diferente"""
def ajustar_valor_al_rango(valor, rango_min, rango_max):
    while valor < rango_min:
        valor += 2 *np.pi
    while valor > rango_max:
        valor -= 2 * np.pi
    return valor

for i in range(len(Phi21)):
    Phi21[i]=ajustar_valor_al_rango(Phi21[i],1,7.3)
        
for i in range(len(Phi31)):
    Phi31[i]=ajustar_valor_al_rango(Phi31[i],4,11.3)
    
for i in range(len(Phi41)):
    Phi41[i]=ajustar_valor_al_rango(Phi41[i],3,9.3)
        
for i in range(len(Phi51)):
    Phi51[i]=ajustar_valor_al_rango(Phi51[i],7,13.3)
        
for i in range(len(Phi61)):
    Phi61[i]=ajustar_valor_al_rango(Phi61[i],5,11.7)

print(V,I,A1,A2,A3,A4,A5,A6,Phi21,Phi31,Phi41,Phi51,Phi61)

# Cambiamos las Phi de serie de cosenos a serie de senos
Phi21s=Phi21-(1/2)*np.pi
Phi31s=Phi31-np.pi
Phi41s=Phi41-(3/2)*np.pi

"""Calculo de [Fe/H]"""
#Relacion de Jurcsik & Kovacs (1996)
FeH_jk=-5.038-5.394*P+1.345*Phi31s

# Se cambia la metalicidad de la escala de Jurcsik a la de Zinn & West segun 
# la ecuacion de Jurcsik (1995) Acta Astronomica
FeHzw=0.6988*FeH_jk-0.6150
# La siguiente ecuación cambia lo valores anteriore a la escala de UVES
FeHuves=-0.413+0.130*FeHzw-0.356*FeHzw**2


# Relacion modificada por NEMEC et al, (2013)ApJ, 773, 181 
# Las Phi31k para esta calibracion estan en el sistema "Kepler" y hay que cambiarlas
# nuestras phi31s en sistema V de Johnson a phi31k usando la relacion siguiente que dan
# Nemec et al. (2013) (ver pg 27 antes de la ec. 3)
# Estas metalicidades estan en al escala de Carretta et al. (2009) o sea UVES
# ver el comentario de Nemec et al. en lineas antes de seccion 5.1

Phi31k=Phi31s+0.151
FeH_nem=-8.65-40.12*P+5.96*Phi31k+6.27*P*Phi31k-0.72*Phi31k**2


# Las lineas que siguen transforman el valor de Nemec et al, en el sistema UVES
# a [Fe/H]zwnem
a=-0.356
b=0.13
c=(-FeH_nem-0.413)
FeHZWnem=(-b+sqrt(b**2-4*a*c))/(2*a)


"""Calculo de Mv"""
# La realción siguiente es la de KOVACS & WALKER (2001)
# Ojo que la constante 0.43 es ya la de Kinman (2002) y la calibracion de Kovacs (2002)
# pero para hacer todo consistente con modulo 18.5 de LMC es necesario adoptar K=0.41
# ve la discusion en nuestro articulo de NGC 5053
Mv=-1.876*np.log10(P)-1.158*A1+.821*A3+.41

"""Cálculo de la Teff"""
# La ecuacion siguiente ES LA DE JURCSIK (1998) PARA LA TEMPERATURA
VK=1.585+1.257*P-0.273*A1-0.234*Phi31s+0.062*Phi41s
Teff=3.9291-0.1112*(VK)-0.0032*FeH_jk

# La ecuacion de Nemec para la temperatura en función de x=(V-I)o
# (ver Arellano Ferro et al 2010 MNRAS 402, 226–244 SECCION 4.3)

VI=V-I #Este valor es el color V-I de cada RR, se calcula restando los A0 de cada filtro
EVI = 1.259*EBV
x=VI-EVI
B0 =+3.9867
B1 =-0.9506 
B2 =+3.5541
B3 =-3.4537
B4 =-26.4992
B5 =+90.9507
B6 =-109.6680
B7 =+46.7704
TNem = B0+B1*x+B2*(x**2)+B3*(x**3)+B4*(x**4)+B5*(x**5)+B6*(x**6)+B7*(x**7)

"""Cálculo de Luminosidad"""
# Aqui se calcula la correccion bolometrica usando la formula de Sandage 
# & Cacciari 1990 que esta calculada para log Teff=3.85. Habiamos usado los 
# valores de Castelli 1999 pero son para FE/H ~ 1.5 para menos metalicos como 
# -1.9 o algo asi usamos la ecuacion de Sandage & Cacciari 1990

# Tenemos la Magitud bolométrica del sol
MBOLSOL=4.75
# A partir del valor de calculado de [Fe/H]_UVES se calcula la corrección bolométrica
BC=0.06*FeHuves+0.06
# Se aplica la correción bolometricaa la magnitud absoluta de la estrella 
MBOL=Mv+BC
# Por último se resta la magnitud bolometrica del col y tenemos la luminosidad en 
# escala logartimica
L=-0.4*(MBOL-MBOLSOL)

"""Cálculo de la Masa"""

# Esta es la ecuacion de van Albada & Baker 1971 y la llamamos MASA(PULSACIONAL)
logM= 16.907-1.47*np.log10(P) + 1.24*L - 5.12*Teff
# Auitamos el logaritmo y tenemos la masa en Masas solares
M=10**(logM)


"""Cálculo de la Distancia"""
# Aqui se calculan los modulos de distancia y se corrigen por extincion.
modulo=V-Mv #V es la magnitud dek filtro V 
truemod=modulo-3.1*EBV
# Calculamos la distancia en parsecs
D=10**((truemod+5)/5)
# La siguiente ecuación es si queremos el valor de de la mag desenrojecido
# Vo=V-3.1*EBV


"""Cálculo del Radio"""
# Temperatura del sol en escala logarítmica
Tsol=3.763
# Comparamos con la temperatura de la RR Lyrae y con la luminosidad, todo sigue
# en escala logarítmica
logR=(L-4*(Teff-Tsol))/2
# Finalmente obtenemos el radio
R=10**(logR)

# Guardamos los datos en una tabla
ParFis_RRab=pd.DataFrame({'[Fe/H]_ZW':FeHzw,'[Fe/H]_UVES':FeHuves,
                          '[Fe/H]_Nem':FeH_nem,'[Fe/H]_ZWNem':FeHZWnem,
                          'Mv':Mv,'log T_eff':Teff,'log (L/Lo)':L,'M/Mo':M,
                          'R/Ro':R,'D(pc)':D})
ParFis_RRab.to_csv("/home/alberto/Escritorio/Modular_Mariano/Ejemplos/ParFis_RRab.txt", sep='\t', index=False)