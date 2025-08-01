#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:17:44 2024

@author: alberto
"""

import matplotlib.pyplot as plt
import pandas as pd

# Leemos los datos de nuestra curva de luz en ambos filtros (V e I)
datosV=pd.read_csv('/home/alberto/Escritorio/Modular_Mariano/Todas/V2_V.dat',
                  header = None, delimiter=(" "))
datosI=pd.read_csv('/home/alberto/Escritorio/Modular_Mariano/Todas/V2_I.dat',
                  header = None, delimiter=(" "))
# Damos el periodo y la epoca de maximo o minimo(eclipsantes) de la variable
P=0.643888 #0.377279  #0.702785969 #0.702786
E0=2458280.8451 #2458280.8720 #2458280.871955127

# Inicializar una lista para almacenar los resultados
faseo = []
"""En estas lineas se itera columna por columna y se le calcula su fase
correspondiente, tomando solo la parte decimal de la operacion, por ende el 
valor de la fase sera entre 0 y 1.

Para que la forma de la curva de luz se muestre completa en su grafica se 
repetiran datos para ciertos rangos de la fase. Para valores mayores a 0.7 y 
menores a 0.3, es decir nuestra grafica tendra un dominio de -0.3 a 1.3"""


# Procesar cada fila del DataFrame, primero con el filtro V
for _, fila in datosV.iterrows():
    HJD = fila[0]
    mag = fila[1]
    fase=(HJD-E0)/P%1
    
    faseo.append([HJD,fase,mag])
    if 0.7 <= fase <= 1.0:
        fase -= 1
        faseo.append([HJD,fase,mag])
    
    if 0.0 <= fase <= 0.3:
        fase += 1
        faseo.append([HJD, fase,mag])
    
"""Guardamos el resultado del faseo"""
# Convertir los resultados a un DataFrame y guardamos 
fasesV = pd.DataFrame(faseo, columns=['HJD','fase','mag'])

"""Hacemos lo mismo para el filtro I """
# Inicializar una lista para almacenar los resultados
faseo = []
# Procesar cada fila del DataFrame
for _, fila in datosI.iterrows():
    HJD = fila[0]
    mag = fila[1]
    fase=(HJD-E0)/P%1
    
    faseo.append([HJD,fase,mag])
    if 0.7 <= fase <= 1.0:
        fase -= 1
        faseo.append([HJD,fase,mag])
    
    if 0.0 <= fase <= 0.3:
        fase += 1
        faseo.append([HJD, fase,mag])
    
"""Guardamos el resultado del faseo"""
# Convertir los resultados a un DataFrame y guardamos
fasesI = pd.DataFrame(faseo, columns=['HJD','fase', 'mag'])


"""Graficamos los puntos de la curva de luz,creamos un subplot 2x1 y graficamos
 por separado para cada filtro, posteriormente le damos formato a la grafica"""
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(6.4, 6.4)) #creamos la grafica 2x1

# Graficamos para el filtro V y cambiamos los parametros de graficado
ax1.scatter(fasesV['fase'],fasesV['mag'],marker='.',c='black')
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
ax2.scatter(fasesI['fase'],fasesI['mag'],marker='.',c='black')
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

# Hacemos los ultimos detalles del formato y guardamos la imagen resultante
plt.rcParams['font.family'] = 'serif'
plt.subplots_adjust(hspace=0) 
plt.savefig('/home/alberto/Escritorio/Modular_Mariano/Ejemplos/V2.png', bbox_inches='tight')
plt.show()

"""En las siguientes lineas guardamos las tablas resultantes del faseo de las 
curvas de luz en los filtros V e I. El HJD se guardara con 9 decimales y la fase
y magnitud con 5, la fase se guarda con signos + y - para que las columnas 
coincidan en formato. Estos archivos .fas seran leidos por el programa Fourier_VI.py
para calcular el ajuste."""

fasesV.apply(lambda col: col.map('{:.9f}'.format) if col.name == 'HJD' 
             else col.map('{:+.5f}'.format) if col.name == 'fase'
             else col.map('{:.5f}'.format)).to_csv(
    '/home/alberto/Escritorio/Modular_Mariano/Ejemplos/V2_V.fas',
    sep='\t',
    index=False,
    header=True
)

fasesI.apply(lambda col: col.map('{:.9f}'.format) if col.name == 'HJD' 
             else col.map('{:+.5f}'.format) if col.name == 'fase'
             else col.map('{:.5f}'.format)).to_csv(
    '/home/alberto/Escritorio/Modular_Mariano/Ejemplos/V2_I.fas',
    sep='\t',
    index=False,
    header=True
)





