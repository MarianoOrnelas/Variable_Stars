#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:10:19 2024

@author: alberto
"""

import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties.unumpy import uarray, sqrt, log10

# Damos el periodo y la epoca de maximo o minimo(eclipsantes)
P=0.377279  #0.702785969 #0.702786
E0=2458279.7663 #2458280.8720 #2458280.871955127
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
    Phi21[i]=ajustar_valor_al_rango(Phi21[i],3,7)
        
for i in range(len(Phi31)):
    Phi31[i]=ajustar_valor_al_rango(Phi31[i],1,5)
    
for i in range(len(Phi41)):
    Phi41[i]=ajustar_valor_al_rango(Phi41[i],0,4)
        
# Cambiamos las Phi de serie de cosenos a serie de senos
Phi21s=Phi21-(1/2)*np.pi
Phi31s=Phi31-np.pi
Phi41s=Phi41-(3/2)*np.pi

"""Calculo de [Fe/H]"""
# Relacion de Morgan S. M., Wahl J. N., Wieckhorst R.M., 2007, MNRAS, 374, 1421
FeHzw=52.466*(P**2)-30.075*P+0.131*(Phi31**2)+0.982*Phi31-4.198*(Phi31*P)+2.424

# La siguiente ecuación cambia lo valores anteriores a la escala de UVES
FeHuves=-0.413+0.130*FeHzw-0.356*FeHzw**2


# Relacion modificada por NEMEC et al, (2013)ApJ, 773, 181 
# Estas metalicidades estan en al escala de Carretta et al. (2009) o sea UVES

FeH_nem=1.70-15.67*P+0.20*Phi31-2.41*(Phi31*P)+18.0*P**2+0.17*(Phi31**2)


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
Mv=1.061 - (0.961*P) -(0.044*Phi21s) - (4.447 * A4)

"""Cálculo de la Teff"""
# la ecuacion siguiente ES LA DE Simon y Clement (1993) PARA LA TEMPERATURA
Teff=3.7746-0.1452*np.log10(P)+0.0056*Phi31 #esta de momento no se usa

# La ecuacion de Nemec para la temperatura en función de x=(V-I)o
# (ver Arellano Ferro et al 2010 MNRAS 402, 226–244 SECCION 4.3)

VI=V-I #Este valor es el color V-I de cada RR
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

ironH=-1.50
# A partir del valor de calculado de [Fe/H]_UVES se calcula la corrección bolométrica
BC=0.06*ironH+0.06
# Se aplica la correción bolometricaa la magnitud absoluta de la estrella 
MBOL=Mv+BC
# Por último se resta la magnitud bolometrica del col y tenemos la luminosidad en 
# escala logartimica
L=-0.4*(MBOL-MBOLSOL)


"""Cálculo de la Masa """
# El periodo en la ecuacion debe ser el del modo fundamental. Asi que para RRc
# su periodo P1 debe ser "fundamentalizado"  P0=P1/0.746 (Cox et al. 1983) 
P0=P/0.746	
# Esta es la ecuacion de van Albada & Baker 1971 y la llamamos MASA(PULSACIONAL)
logM= 16.907-1.47*log10(P0) + 1.24*L- 5.12*TNem
# Auitamos el logaritmo y tenemos la masa en Masas solares
M=10**(logM)


"""Cálculo de la Distancia"""
# Aqui se calculan los modulos de distancia y se corrigen por extincion.
modulo=V-Mv
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
logR=(L-4*(TNem-Tsol))/2
# Finalmente obtenemos el radio
R=10**(logR)

# Guardamos los datos en una tabla
ParFis_RRc=pd.DataFrame({'[Fe/H]_ZW':FeHzw,'[Fe/H]_UVES':FeHuves,
                          '[Fe/H]_Nem':FeH_nem,'[Fe/H]_ZWNem':FeHZWnem,
                          'Mv':Mv,'log T_eff':TNem,'log (L/Lo)':L,'M/Mo':M,
                          'R/Ro':R,'D(pc)':D})
ParFis_RRc.to_csv("ParFis_RRc.txt", sep='\t', index=False)
ParFis_RRc.to_csv("/home/alberto/Escritorio/Modular_Mariano/Ejemplos/ParFis_RRc.txt", sep='\t', index=False)