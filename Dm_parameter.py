#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:39:20 2024

@author: alberto
"""

import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy
"""Leemos los coeficiones de fourier y les damos el formato adecuado para que lea
el valor y su error"""
coef_Fou = pd.read_csv('/home/alberto/Escritorio/Modular_Mariano/Ejemplos/Salida.dat', delimiter=("\t"))

def to_ufloat(text):
    return ufloat(*map(float, text.split('+/-')))

for column in coef_Fou.columns[1:]:
    coef_Fou[column] = coef_Fou[column].apply(to_ufloat)
    #print(column)

for column_name in coef_Fou.columns[1:]:
    globals()[column_name] = unumpy.nominal_values(coef_Fou[column_name].values)

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

print(A1,A2,A3,A4,A5,A6,Phi21,Phi31,Phi41,Phi51,Phi61)
# Convertimos a Phi en serie de senos, los que tenemos calculado son para serie de cos
Phi21s=Phi21-(1/2)*np.pi
Phi31s=Phi31-np.pi
Phi41s=Phi41-(3/2)*np.pi
Phi51s=Phi51-2*np.pi
Phi61s=Phi61-(5/2)*np.pi

#Las que siguen son las formulas de interelacion de parametros de Fourier 
#Kovac & Kanbur (1998) para calcular F_calc (ver su Tabla 2)

A1C=0.611+1.406*A2-0.099*Phi31s
A2C=-0.289+0.379*A1+0.580*A3+0.051*Phi31s
Phi21C=-0.296+0.991*A1-2.047*A3+0.507*Phi31s
A3C=0.011+0.229*A2+0.886*A4
Phi31C=3.708+0.347*Phi21s+0.728*Phi41s-0.137*Phi51s
A4C=-0.001+0.369*A3+0.689*A5
Phi41C=-4.019+0.837*Phi31s+0.306*Phi51s
A5C=-0.002+0.381*A4+0.710*A6
Phi51C=2.673+0.924*Phi41s+0.278*Phi61s
 
#En lo que sigue se calculan los parametros Dm para cada parametro de Fourier

DA1=abs(A1-A1C)/0.0211
DA2=abs(A2-A2C)/0.0118
DPhi21=abs(Phi21s-Phi21C)/0.0615
DA3=abs(A3-A3C)/0.0071
DPhi31=abs(Phi31s-Phi31C)/0.0576
DA4=abs(A4-A4C)/0.0046
DPhi41=abs(Phi41s-Phi41C)/0.0617
DA5=abs(A5-A5C)/0.0027
DPhi51=abs(Phi51s-Phi51C)/0.1221

# Creamos un data frame y guardamos

Dm=pd.DataFrame({'DA1':DA1,'DA2':DA2,'DA3':DA3,'DA4':DA4,'DA5':DA5,
                 'DPhi21':DPhi21,'DPhi31':DPhi31,'DPhi41':DPhi41,
                 'DPhi51':DPhi51})

Dm.to_csv('/home/alberto/Escritorio/Modular_Mariano/Ejemplos/Dm.dat'
                 , index=False, sep='\t', header=True)