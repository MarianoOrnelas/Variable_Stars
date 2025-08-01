#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 15:46:52 2025

@author: ces
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:39:20 2025

@author: ces
"""



import numpy as np
import pandas as pd
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
from uncertainties.unumpy import uarray, sqrt, log10, arctan2, nominal_values, std_devs
import matplotlib.pyplot as plt
import sys
from io import StringIO
import os
from pathlib import Path
import re
from adjustText import adjust_text


def pausa():
    """Pausa la ejecución y pregunta si el usuario desea continuar."""
    respuesta = input("\n¿Deseas continuar con la siguiente sección? (s/n): ").strip().lower()
    if respuesta != 's':
        print("Ejecución detenida")
        sys.exit()
        
        
def obtener_E0(rutaI, rutaV, Type):
    """
    Encuentra el HJD más extremo entre dos archivos de magnitudes (filtros I y V)
    
    Parámetros:
    rutaI (str): Ruta al archivo de datos del filtro I
    rutaV (str): Ruta al archivo de datos del filtro V
    Type (str): Tipo de variable ('ECL', 'EW', etc. para máximos, otros para mínimos)
    
    Retorna:
    float: HJD correspondiente al evento más extremo
    """
    # Función para leer y limpiar datos
    def leer_limpiar(ruta):
        df = pd.read_csv(
            ruta,
            delim_whitespace=True,
            comment='#',
            usecols=[0, 1],
            names=['HJD', 'mag'],
            header=0
        )
        df['mag'] = pd.to_numeric(df['mag'], errors='coerce')
        return df.dropna(subset=['mag'])
    
    # Leer y procesar ambos archivos
    dfI, dfV = map(leer_limpiar, [rutaI, rutaV])
    
    # Verificar datos
    if dfI.empty or dfV.empty:
        raise ValueError("Datos insuficientes después de limpieza")
    
    # Determinar operación (máximo para variables eclipsantes, mínimo para otras)
    buscar_max = Type.upper() in {"ECL", "EW", "EA", "EB", "EA/EB"}
    extremo = 'max' if buscar_max else 'min'
    
    # Encontrar valores extremos y sus índices
    valI, idxI = getattr(dfI['mag'], extremo)(), getattr(dfI['mag'], f'idx{extremo}')()
    valV, idxV = getattr(dfV['mag'], extremo)(), getattr(dfV['mag'], f'idx{extremo}')()
    
    # Seleccionar el HJD más extremo
    if buscar_max:
        return dfI.loc[idxI, 'HJD'] if valI >= valV else dfV.loc[idxV, 'HJD']
    return dfI.loc[idxI, 'HJD'] if valI <= valV else dfV.loc[idxV, 'HJD']
        

def ajustar_valor_al_rango(valor, rango_min, rango_max):
    while valor < rango_min:
        valor += 2 *np.pi
    while valor > rango_max:
        valor -= 2 * np.pi
    return valor
       

def fasear(star,Type,StarType, ruta, P,E0, n_terms, prin_group, guardar, filtro, ajuste=False):


    datos = pd.read_csv(ruta, delimiter="\s+",header=None)
    
    if datos.iloc[:, 0].astype(float).mean() < 2450000:
        datos.iloc[:, 0] += 2450000

    faseo = []

    saltar_faseo = (StarType == "Mira") or (StarType == "SR" and P == 0)

    for _, fila in datos.iterrows():
        HJD = fila.iloc[0]
        mag = fila.iloc[1]
        ver = fila.iloc[-1]

        if saltar_faseo:
            faseo.append([HJD, mag, ver])
        else:
            fase = (HJD - E0) / P % 1
            faseo.append([HJD, fase, mag, ver])
            if 0.7 <= fase <= 1.0:
                fase -= 1
                faseo.append([HJD, fase, mag, ver])
            if 0.0 <= fase <= 0.3:
                fase += 1
                faseo.append([HJD, fase, mag, ver])

    if saltar_faseo:
        fases = pd.DataFrame(faseo, columns=['HJD', 'mag', 'ver'])
    else:
        fases = pd.DataFrame(faseo, columns=['HJD', 'fase', 'mag', 'ver'])
        fases = fases.sort_values(by="fase")

    if guardar == "s":
        fases.to_csv(f"{ruta_Datafas}/{filtro}/{star}.fas", sep="\t")

    if not ajuste:
        return fases

    aj = []
    LogAjusteGrupo = {"ID": [], "filtro": [], "conjunto": [], "ajustado": [], "ajuste": [], "razon": []}
    n_data = int(fases["ver"].max())

    for i in range(1, n_data + 1):
        conjunto = fases[fases["ver"] == i]

        if conjunto.shape[0] < (2 * n_terms + 1):
            aj.append(0)
            if i != prin_group:
                LogAjusteGrupo["ID"].append(star)
                LogAjusteGrupo["filtro"].append(filtro)
                LogAjusteGrupo["conjunto"].append(i)
                LogAjusteGrupo["ajustado"].append(False)
                LogAjusteGrupo["ajuste"].append(np.nan)
                LogAjusteGrupo["razon"].append("pocos_datos")
            continue

        HJD = conjunto["HJD"]
        mag = conjunto["mag"]

        if saltar_faseo:
            xdata = HJD
        else:
            fase = conjunto["fase"]
            if fase.diff().gt(0.17).any():
                aj.append(0)
                if i != prin_group:
                    LogAjusteGrupo["ID"].append(star)
                    LogAjusteGrupo["filtro"].append(filtro)
                    LogAjusteGrupo["conjunto"].append(i)
                    LogAjusteGrupo["ajustado"].append(False)
                    LogAjusteGrupo["ajuste"].append(np.nan)
                    LogAjusteGrupo["razon"].append("discontinuidad_fase")
                continue
            xdata = HJD  # modelo aún usa HJD como variable

        m = fourier_fun(n_terms, P, E0)
        p0 = [1] * (1 + 2 * n_terms)
        bounds = ([-np.inf] * len(p0), [np.inf] * len(p0))

        try:
            res, _ = curve_fit(m, xdata, mag, p0=p0, bounds=bounds)
            aj.append(res[0])
        except Exception as e:
            aj.append(0)
            if i != prin_group:
                LogAjusteGrupo["ID"].append(star)
                LogAjusteGrupo["filtro"].append(filtro)
                LogAjusteGrupo["conjunto"].append(i)
                LogAjusteGrupo["ajustado"].append(False)
                LogAjusteGrupo["ajuste"].append(np.nan)
                LogAjusteGrupo["razon"].append("error_curve_fit")

    if prin_group > len(aj) or aj[prin_group - 1] == 0:
        for i, j in enumerate(aj):
            if i + 1 != prin_group:
                LogAjusteGrupo["ID"].append(star)
                LogAjusteGrupo["filtro"].append(filtro)
                LogAjusteGrupo["conjunto"].append(i + 1)
                LogAjusteGrupo["ajustado"].append(False)
                LogAjusteGrupo["ajuste"].append(np.nan)
                LogAjusteGrupo["razon"].append("grupo_referencia_sin_ajuste")
    else:
        for i, j in enumerate(aj):
            if j == 0 or i + 1 == prin_group:
                continue
            delta = aj[prin_group - 1] - aj[i]
            suma_por_grupo = {i + 1: delta}
            print(suma_por_grupo)

            LogAjusteGrupo["ID"].append(star)
            LogAjusteGrupo["filtro"].append(filtro)
            LogAjusteGrupo["conjunto"].append(i + 1)
            LogAjusteGrupo["ajustado"].append(True)
            LogAjusteGrupo["ajuste"].append(delta)
            LogAjusteGrupo["razon"].append("ajuste_exitoso")

            fases['mag'] = fases.apply(lambda row: row['mag'] + suma_por_grupo.get(row['ver'], 0), axis=1)
            datos[1] = datos.apply(lambda row: row[1] + suma_por_grupo.get(row[1], 0), axis=1)



    LogAjusteGrupo = pd.DataFrame(LogAjusteGrupo)

    if guardar == "s":
        fases.to_csv(f"{ruta_Datafas}/{filtro}/{star}.fas",index=None, sep="\t",header=False)
        datos.to_csv(f"{ruta_Datafit}/{filtro}/{star}.dat",index=None, sep="\t",header=False)

    return fases, LogAjusteGrupo



    
def subplotfas(DF_faseado, StarType, ax, filtro, xlim_common=None):
    """
    Esta función genera la gráfica del faseo o de HJD vs magnitud para uno de los filtros,
    dependiendo del tipo de estrella.
    """
    global n_data
    n_data = int(DF_faseado["ver"].max())
    colors = plt.cm.get_cmap("rainbow", n_data)

    if StarType in ["Mira", "SR_0"]:
        xvar = "HJD"
    else:
        xvar = "fase"

    for k in range(1, n_data + 1):
        subset = DF_faseado[DF_faseado["ver"] == k]
        ax.plot(subset[xvar], subset["mag"], ".", color=colors(k - 1), markersize=1)

    ax.invert_yaxis()
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
    ax.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
    ax.set_ylabel(filtro, fontstyle="italic", fontsize=14)

    if xvar == "fase" and xlim_common is not None:
        ax.set_xlim(xlim_common)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([])  # Solo se etiqueta abajo
    elif xvar == "fase":
        xlim = [DF_faseado[xvar].min(), DF_faseado[xvar].max()]
        ax.set_xlim([xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0])])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([])

    elif xvar == "HJD":
        xlim = [DF_faseado[xvar].min(), DF_faseado[xvar].max()]
        ax.set_xlim([xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0])])
        NUM = max(DF_faseado["HJD"]) - min(DF_faseado["HJD"])
        ticks = [round(min(DF_faseado["HJD"]) + NUM / 4 * n, -1) for n in range(1, 6)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([])

    if StarType in ["Mira", "SR_0"]:
        ymin = DF_faseado["mag"].min()
        ymax = DF_faseado["mag"].max()
        ax.set_ylim([ymin - 1, ymax + 1])
    else:
        ylim = [DF_faseado["mag"].min(), DF_faseado["mag"].max()]
        ax.set_ylim([ylim[0] - 0.1, ylim[1] + 0.1])

    return ax


def plot_fas(fasesI, fasesV, star, StarType, guardar):
    """
    Esta función usa los subplots generados con 'subplotfas' para generar la gráfica completa.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.4))  # Creamos la gráfica 2x1

    if StarType not in ["Mira", "SR_0"]:
        xlim_common = [-0.3, 1.3]
    else:
        xlim_common = None

    ax1 = subplotfas(fasesV, StarType, ax1, "V", xlim_common)
    ax2 = subplotfas(fasesI, StarType, ax2, "I", xlim_common)

    # Etiqueta de eje x
    if StarType in ["Mira", "SR_0"]:
        if max(fasesV["HJD"]) - min(fasesV["HJD"]) > max(fasesI["HJD"]) - min(fasesI["HJD"]):
            NUM = max(fasesV["HJD"]) - min(fasesV["HJD"])
            ticks = [f"{round(NUM / 4 * n, -1)}" for n in range(1, 5)] + [""]
            ax2.set_xticklabels(ticks, fontsize=12)
            ax2.set_xlabel("HJD (2 450 000+)", fontsize=14)
        else:
            NUM = max(fasesI["HJD"]) - min(fasesI["HJD"])
            ticks = [f"{round(NUM / 4 * n, -1)}" for n in range(1, 5)] + [""]
            ax2.set_xticklabels(ticks, fontsize=12)
            ax2.set_xlabel("HJD (2 450 000+)", fontsize=14)
    else:
        ax2.set_xlabel(r"$\phi$", fontsize=14)
        ax2.set_xticks([0, 0.5, 1])
        ax2.set_xticklabels(["0", "0.5", "1"], fontsize=12)

    if star.startswith("OGLE"):
        num = re.search(r'\d+$', star).group(0)
    else:
        num = star

    plt.rcParams['font.family'] = 'serif'
    plt.subplots_adjust(hspace=0)
    plt.suptitle(num, fontsize=16)
    plt.subplots_adjust(top=1.005)
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    if guardar == "s":
        plt.savefig(f"{ruta_faseadas}/{star}.pdf", bbox_inches='tight')

    return fig, ax1, ax2


        
def fourier_fun(n_terms, P, E0):
    def m(x, *params):
        A0 = params[0]  # Término constante
        result = A0
        for n in range(1, n_terms + 1):
            An = params[n]  # Coeficiente del coseno
            Bn = params[n_terms + n]  # Coeficiente del seno
            result += An * np.cos(n * (2 * np.pi / P) * (x - E0)) + Bn * np.sin(n * (2 * np.pi / P) * (x - E0))
        return result
    return m

def conv_a_cos(params,n_terms):
    a = params[1:n_terms+1]  # Coeficientes del coseno
    b = params[n_terms+1:len(coef)] #coeficientes de seno
    phi = []  # Fases
    
    A=sqrt(a**2 + b**2) # Se calcula A que lo tenias como C
    A=np.append(params[0],A) # se agrega de forma correcta el valor A[0]
    phi= arctan2(-b, a)  # Usamos arctan2 de uncertainties.unumpy para manejar correctamente los cuadrantes

    return A, phi


ruta=Path(__file__)
carpeta= str(ruta.resolve().parent)



manystars=input("¿Quieres hacer una sola estrella?, en el caso contrario se haran todas (s/n) ").strip().lower()
raw=input(" Quieres usar los datos ajustados? (de la carpeta fit) (s/n) ").strip().lower()
if manystars == "s":
    OneStar=input("Inserta el ID de la estrella que quieres hacer ").strip()
if raw != "s":
    calcE0=input("¿Quieres calcular el E0 (s) o usar el de la fst?(n) ").strip().lower()
    ajuste0=input("¿Quieres que se haga ajuste al punto 0 para los datos? ").strip().lower()
    if ajuste0 == "s":
        ajuste0=True
        n_data=int(input("Cuantos grupos de datos ser estan usando? "))
        prin_group=int(input("¿Cual es tu grupo principal de datos? "))
    else:
        ajuste0=False 
        prin_group=0
else:
    calcE0="n"
    prin_group=0
    ajuste0=False 
    
guardar=input("¿Deseas guardar las graficas, tablas y datos? (s/n) ").strip().lower()


fst = pd.read_csv(f"{carpeta}/tablas/fst_test.txt",sep='\s+')


max_arm=int(fst[["Arms_I","Arms_V"]].values.max())



# fst = fst[(fst['Type'] == "RRLyr") ]

if manystars == "s":
    fst=fst[fst["ID"]==OneStar]



# Nos aseguramos de que el tiempo este en el formato adecuado
if fst['T0_1'].astype(float).mean()< 2450000:
    fst['T0_1']=fst['T0_1'].astype(float)+2450000

# definimos el dataframe en donde se guardaran los parametros fisicos
ParFisAB=pd.DataFrame(columns=('ID','[Fe/H]_ZWNem','[Fe/H]_Nem',
                             '[Fe/H]_ZW','[Fe/H]_UVES',
                          'Mv','log T_eff','log (L/Lo)','M/Mo',
                          'R/Ro','D(pc)'))
ParFisAB_noDm=pd.DataFrame(columns=('ID','[Fe/H]_ZWNem','[Fe/H]_Nem',
                             '[Fe/H]_ZW','[Fe/H]_UVES',
                          'Mv','log T_eff','log (L/Lo)','M/Mo',
                          'R/Ro','D(pc)'))
ParFisC=pd.DataFrame(columns=('ID','[Fe/H]_ZWNem','[Fe/H]_Nem',
                             '[Fe/H]_ZW','[Fe/H]_UVES',
                          'Mv','log T_eff','log (L/Lo)','M/Mo',
                          'R/Ro','D(pc)'))
salida_out=pd.DataFrame(columns=['ID','V','I','V-I','A1','A2','A3','A4','A5','A6',
                                         'Phi21','Phi31','Phi41','Phi51','Phi61'])

Arm_fou = pd.DataFrame(columns=
    ["ID"]+[f"a{i}" for i in range(0, max_arm+1)] + [f"b{i}" for i in range(1, max_arm+1)]
    +['R21','R31','R41','R32','R42','R43'])    
Fou_params=pd.DataFrame(columns=['ID','A0','A1','A2','A3','A4','A5','A6',
                                     'Phi21','Phi31','Phi41','Phi51','Phi61'])
Tabla_General=pd.DataFrame(columns=["ID","Type","Subtype","I","V","A_v","A_i","P","HJD_max"])


Dm_out=pd.DataFrame(columns=['ID','DA1','DA2','DA3','DA4','DA5',
                     'DPhi21','DPhi31','DPhi41',
                     'DPhi51'])
NoAptas=pd.DataFrame(columns=['ID'])
LogAjuste=pd.DataFrame(columns=["ID","filtro","conjunto","ajustado","razon"])
NoFiles=pd.DataFrame(columns=["ID","Filtro"])
A_vi=pd.DataFrame(columns=["ID","Subtype","A_v","A_i","Max_V","Min_V","Max_I","Min_I"])

if guardar == "s":
    ruta_log=f"{carpeta}/log"
    ruta_graficos=f"{carpeta}/graficos"
    ruta_faseadas=f"{carpeta}/graficos/faseadas"
    ruta_fourier=f"{carpeta}/graficos/fourier"
    ruta_salida=f"{carpeta}/Salida"
    ruta_Datafit=f"{carpeta}/Data_fit"
    ruta_Datafit_I=f"{carpeta}/Data_fit/I"
    ruta_Datafit_V=f"{carpeta}/Data_fit/V"
    ruta_Datafas=f"{carpeta}/Data_fas"
    ruta_Datafas_I=f"{carpeta}/Data_fas/I"
    ruta_Datafas_V=f"{carpeta}/Data_fas/V"    
    
    Path(ruta_log).mkdir(parents=True, exist_ok=True)
    Path(ruta_graficos).mkdir(parents=True, exist_ok=True)
    Path(ruta_faseadas).mkdir(parents=True, exist_ok=True)
    Path(ruta_fourier).mkdir(parents=True, exist_ok=True)
    Path(ruta_salida).mkdir(parents=True, exist_ok=True)
    
    Path(ruta_Datafit).mkdir(parents=True, exist_ok=True)
    Path(ruta_Datafit_I).mkdir(parents=True, exist_ok=True)
    Path(ruta_Datafit_V).mkdir(parents=True, exist_ok=True)
    
    Path(ruta_Datafas).mkdir(parents=True, exist_ok=True)
    Path(ruta_Datafas_I).mkdir(parents=True, exist_ok=True)
    Path(ruta_Datafas_V).mkdir(parents=True, exist_ok=True)

for star in fst['ID']:
    x=fst[(fst['ID']== star)]
    StarType=str(x["Subtype"].iloc[0])
    Type=str(x["Type"].iloc[0])
    n_terms_I=int(x["Arms_I"].iloc[0])
    n_terms_V=int(x["Arms_V"].iloc[0])
    if n_terms_I == 0:
        n_terms_I+=1
    if n_terms_V==0:
        n_terms_V+=1
    
    if Type == "SR":
        StarType== "SR"

    EBV= .02    
    
    """Damos la ruta a nuestros archivos con la fotometria"""
    rutaI=f'{carpeta}/DATA_RAW/I/{star}.dat'
    rutaV=f'{carpeta}/DATA_RAW/V/{star}.dat'
    if raw== "s":
        rutaI=f'{carpeta}/Data_fit/I/{star}.dat'
        rutaV=f'{carpeta}/Data_fit/V/{star}.dat'
    else:
        rutaI=f'{carpeta}/DATA_RAW/I/{star}.dat'
        rutaV=f'{carpeta}/DATA_RAW/V/{star}.dat'
    
    
    if not Path(rutaI).is_file() and not Path(rutaV).is_file() :
        NoFile=pd.DataFrame({"ID":[star],"Filtro":["I,V"]})
        NoFiles=pd.concat([NoFiles,NoFile],ignore_index=True)
        continue
    elif Path(rutaV).is_file() and not Path(rutaI).is_file():
        dataI=False
        dataV=True
        NoFile=pd.DataFrame({"ID":[star],"Filtro":["I"]})
        NoFiles=pd.concat([NoFiles,NoFile],ignore_index=True)
    elif Path(rutaI).is_file() and not Path(rutaV).is_file():
        dataV=False
        dataI=True
        NoFile=pd.DataFrame({"ID":[star],"Filtro":["V"]})
        NoFiles=pd.concat([NoFiles,NoFile],ignore_index=True)
    else:
        dataI=True
        dataV=True
    # Damos el periodo y la epoca de maximo o minimo(eclipsantes) de la variable
    P=float(x['P_1'].iloc[0])
    if calcE0 =="n":
        E0=float(x['T0_1'].iloc[0])
    else:
        E0=obtener_E0(rutaI, rutaV, Type)
    if E0 < 2450000:
        E0=E0+2450000
         
    if dataI== False:
        fasesI=pd.DataFrame({"HJD":[0],"fase":[-1],"mag":[0],"ver":[prin_group]})
    else:
        if ajuste0 == True:
            fasesI,logAjusteGrupoI =fasear(star,Type,StarType,rutaI,P,E0,n_terms_I,prin_group,guardar,"I",ajuste=True)
        else:
            fasesI=fasear(star,Type,StarType,rutaI,P,E0,n_terms_I,prin_group,guardar,"I",ajuste=False)

    if dataV== False:
        fasesV=pd.DataFrame({"HJD":[0],"fase":[-1],"mag":[0],"ver":[prin_group]})
    else:
        if ajuste0== True:
             fasesV,logAjusteGrupoV=fasear(star,Type,StarType,rutaV,P,E0,n_terms_V,prin_group,guardar,"V",ajuste=True)
        else:
            fasesV=fasear(star,Type,StarType,rutaV,P,E0,n_terms_V,prin_group,guardar,"V",ajuste=False)

    if ajuste0==True:
        if (dataI and dataV)== True:
            LogAjuste=pd.concat([LogAjuste,logAjusteGrupoI,logAjusteGrupoV])
        elif dataI ==True and dataV==False:
            LogAjuste=pd.concat([LogAjuste,logAjusteGrupoI])
        elif dataV ==True and dataI==False:
            LogAjuste=pd.concat([LogAjuste,logAjusteGrupoV])
    
# Este se DF se usa mas adelante para el diagrama de Beilis
    a_vi = pd.DataFrame([[star, StarType,
      max(fasesV["mag"]) - min(fasesV["mag"]), 
      max(fasesI["mag"]) - min(fasesI["mag"]), 
      max(fasesV["mag"]), 
      min(fasesV["mag"]), 
      max(fasesI["mag"]), 
      min(fasesI["mag"])]],
    columns=["ID","Subtype", "A_v", "A_i", "Max_V", "Min_V", "Max_I", "Min_I"])
    
    A_vi=pd.concat([A_vi,a_vi])
    
# =========== Se hace la grafica de la curva faseada =====================    

    if (P == 0 and StarType == "SR"):
        StarType == "SR_0"
    fig,ax1,ax2=plot_fas(fasesI, fasesV, star,StarType,guardar)
    
    if StarType in ["Mira", "SR_0"]:
        plt.show()
        continue
    
# =========== A partir de aqui se hace el ajuste de Fourier ==============
    x = np.linspace(-0.3, 1.3, 321)
    t = x * P + E0
    
    if fasesI.shape[0] < (2 * n_terms_I + 1):
        NoApta = pd.DataFrame({"ID": [star]})
        NoAptas = pd.concat([NoAptas, NoApta], ignore_index=True)
    else:
        HJDI = fasesI["HJD"]
        magI = fasesI['mag']
        
        m_I = fourier_fun(n_terms_I, P, E0)
        # Parámetros iniciales y límites
        p0_I = [1] * (1 + 2 * n_terms_I)
        bounds_lower_I = [-np.inf] * (1 + 2 * n_terms_I)
        bounds_upper_I = [np.inf] * (1 + 2 * n_terms_I)
        
        resI, covI = curve_fit(m_I, HJDI, magI, p0=p0_I, bounds=(bounds_lower_I, bounds_upper_I))
        
        Iaj = m_I(t, *resI)
        ajusteI = pd.DataFrame({'fase': x, 'mag': Iaj})
        ax2.plot(x, Iaj, color='black')
        
        
    
    if fasesV.shape[0] < (2 * n_terms_V + 1):
        NoApta = pd.DataFrame({"ID": [star]})
        NoAptas = pd.concat([NoAptas, NoApta], ignore_index=True)
                
    else:
        HJDV = fasesV["HJD"]
        magV = fasesV['mag']
        m_V= fourier_fun(n_terms_V, P, E0)
        p0_V = [1] * (1 + 2 * n_terms_V)
        bounds_lower_V = [-np.inf] * (1 + 2 * n_terms_V)
        bounds_upper_V = [np.inf] * (1 + 2 * n_terms_V)
        resV, covV = curve_fit(m_V, HJDV, magV, p0=p0_V, bounds=(bounds_lower_V, bounds_upper_V))
        Vaj = m_V(t, *resV)
        ajusteV = pd.DataFrame({'fase': x, 'mag': Vaj})
        ax1.plot(x, Vaj, color='black')


    if guardar =="s":
        plt.savefig(f"{ruta_fourier}/{star}.pdf", bbox_inches='tight')
    plt.show()
    """Vamos a guardar los valores de A y phi del resultado del ajuste de Fourier"""
    
    error=np.sqrt(np.diag(covV)) # Error de a partir de la diagonal de la covarianza
    coef=unumpy.uarray(resV,error) # Resultados con sus errores correspondientes

    """
    Si estamos usando menos de 6 armonicos se insertan valores a las tablas y posteriormente se elimina
    en el calculo del DM
    """
    

    a0= coef[0]
    an = coef[1:n_terms_V+1]  
    bn = coef[n_terms_V + 1:]  
    
    #Esto es para que el DF con todos los armonicos tenga la misma dimension 
    relleno = [0]*(max_arm - len(an))

    arm_fou1 = pd.DataFrame(
        [[star]+[a0] + list(an)+relleno+list(bn) + relleno],
        columns=["ID"]+[f"a{i}" for i in range(0, max_arm+1)] + [f"b{i}" for i in range(1, max_arm+1)])
        
    Amps,phises=conv_a_cos(coef, n_terms_V)
    A=Amps
    phi=phises
    while len(A)<7:
        a=ufloat(0,000000)
        A=np.append(A,[a])
        phi=np.append(phi,[a])
        
    """
    Las siguientes iteraciones son para calcular los coeficientes de fourier: 
    Rij=Ai/Aj y Phiij=j*phi[i-1]-i*phi[j-1] a partir de los valores de A y phi.
    Para ello, separamos las amplitudes y los armonicos
    """
    
    
    """A0 le podemos llamar intensity weigthed mean, con notacion <V> o <I>, 
    segun el filtro"""
    V=A[0] # Guardamos A0
    if fasesI.shape[0] < (2 * n_terms_V + 1):
        I=0
    else:
        I=ufloat(resI[0],np.sqrt(covI[0,0]))
    
    # Realizamos las iteraciones
    for i in [2,3,4]:
        for j in [1,2,3]:
            nombre_variable = f"R{i}{j}"  # Crear el nombre de la variable
            valor = A[i]/A[j]  # Calculamos el valor
            globals()[nombre_variable] = valor  # Guardamos el valor como variable global
    
    
    for i in [2,3,4,5,6]:
        for j in [1,2,3]:
            nombre_variable = f"Phi{i}{j}" # Crear el nombre de la variable
            valor=j*phi[i-1]-i*phi[j-1]  # Calculamos el valor
            valor=ajustar_valor_al_rango(valor,0,2*np.pi) # El valor siempre sera ente 0 y 2pi
            globals()[nombre_variable] = valor # Guardamos el valor como variable global
           
    # Guardamos los coeficientes de Fourier 
    CF=[R21,R31,R41,R32,R42,R43]
    arm_fou2=pd.DataFrame([CF],columns=['R21','R31','R41','R32','R42','R43'])
    arm_fou=pd.concat([arm_fou1,arm_fou2],axis=1)
    Arm_fou=pd.concat([Arm_fou,arm_fou],axis=0)

    params= [star, V, A[1],A[2],A[3],A[4],A[5],A[6],Phi21,Phi31,Phi41,Phi51,Phi61]
    fou_params=pd.DataFrame([params],columns=['ID','A0','A1','A2','A3','A4','A5','A6',
                                         'Phi21','Phi31','Phi41','Phi51','Phi61'])
    Fou_params=pd.concat([Fou_params,fou_params],axis=0)
    
    sal=[star,V,I,V-I,A[1],A[2],A[3],A[4],A[5],A[6],Phi21,Phi31,Phi41,Phi51,Phi61]
    salida=pd.DataFrame([sal],columns=['ID','V','I','V-I','A1','A2','A3','A4','A5','A6',
                                         'Phi21','Phi31','Phi41','Phi51','Phi61'])
    salida_out=pd.concat([salida_out,salida],ignore_index=True)
    
    tabla_General = pd.DataFrame(
        [[star, Type, StarType, I, V, round(max(fasesV["mag"]) - min(fasesV["mag"]),4), 
          round(max(fasesI["mag"]) - min(fasesI["mag"]),4), P, E0]],
        columns=["ID", "Type", "Subtype", "I", "V", "A_v", "A_i", "P", "HJD_max"])

    Tabla_General=pd.concat([Tabla_General,tabla_General],ignore_index=True)

# ============ Se hace el analisis para las RR Lyrae ==========================
    
    if StarType=="RRab" or StarType=="BL Her" :  
        coef_Fou = salida
        
        """Primero obtenemos los ParFis sin importar el Dm"""
        for column_name in coef_Fou.columns[1:]:
            globals()[column_name] = coef_Fou[column_name].values
        
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
        """En esta parte puede suceder que las metalicidades sean muy cercanas al 0,
        por lo que las raices pueden ser complejas y dar error, para eso implementamos que sea 0
        en caso de dar error"""
        try: 
            FeHZWnem=(-b+sqrt(b**2-4*a*c))/(2*a)
        except Exception as e:
            FeHZWnem=0
            

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
        ParFis_RRab=pd.DataFrame({'ID':star,'[Fe/H]_ZWNem':FeHZWnem,'[Fe/H]_Nem':FeH_nem,
                                  '[Fe/H]_ZW':FeHzw,'[Fe/H]_UVES':FeHuves,
                                  'Mv':Mv,'log T_eff':Teff,'log (L/Lo)':L,'M/Mo':M,
                                  'R/Ro':R,'D(pc)':D})
        
        ParFisAB_noDm=pd.concat([ParFisAB_noDm,ParFis_RRab],ignore_index=True)
        
        """Ahora si calculamos el Dm y posteriormente los parametros fisicos"""
        
        for column_name in coef_Fou.columns[1:]:
            globals()[column_name] = nominal_values(coef_Fou[column_name])
           
        
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
        

        if n_terms_V == 5:
            DA5=0
            DPhi51=0
        elif n_terms_V==4:
            DA4=0
            DA5=0
            DPhi31=0
            DPhi41=0
            DPhi51=0

        
        # Creamos un data frame y guardamos
        
        Dm_tot=pd.DataFrame({'ID':star,'DA1':DA1,'DA2':DA2,'DA3':DA3,'DA4':DA4,'DA5':DA5,
                         'DPhi21':DPhi21,'DPhi31':DPhi31,'DPhi41':DPhi41,
                         'DPhi51':DPhi51})
        Dm_yes=pd.DataFrame({'DA1':DA1,'DA2':DA2,'DA3':DA3,'DA4':DA4,'DA5':DA5,
                         'DPhi21':DPhi21,'DPhi31':DPhi31,'DPhi41':DPhi41,
                         'DPhi51':DPhi51})
        Dm_out=pd.concat([Dm_out,Dm_tot],ignore_index=True)

        """
        A continuacion tomamos diferentes casos dependiendo de cuantos armonicos hayamos
        utilizado, solo se eliminan las columnas en donde hayamos usado valores extremos
        
        """
        
        if n_terms_V>= 6:
            Dm=Dm_yes
            print("Se estan considerando solo 6 terminos")
        
        elif n_terms_V == 5:
            Dm=Dm_tot.drop("DA5",axis=1).drop("DPhi51",axis=1)
            print("Solo se estan usando 5 armonicos")
            Dm=Dm.drop("ID",axis=1)
        elif n_terms_V==4:
            Dm=Dm_tot.drop("DA4",axis=1).drop("DA5",axis=1).drop("DPhi31",axis=1).drop("DPhi41",axis=1).drop("DPhi51",axis=1)
            print("solo se estan usando 4 armonicos")
            Dm=Dm.drop("ID",axis=1)
        if (Dm.iloc[0]<5).all() == True:
            print('apta para analisis') 
            
            for column_name in coef_Fou.columns[1:]:
                globals()[column_name] = coef_Fou[column_name].values
            
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
            try: 
                FeHZWnem=(-b+sqrt(b**2-4*a*c))/(2*a)
            except Exception as e:
                FeHZWnem=0
    
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
            ParFis_RRab=pd.DataFrame({'ID':star,'[Fe/H]_ZWNem':FeHZWnem,'[Fe/H]_Nem':FeH_nem,
                                      '[Fe/H]_ZW':FeHzw,'[Fe/H]_UVES':FeHuves,
                                      'Mv':Mv,'log T_eff':Teff,'log (L/Lo)':L,'M/Mo':M,
                                      'R/Ro':R,'D(pc)':D})
            
            ParFisAB=pd.concat([ParFisAB,ParFis_RRab],ignore_index=True)
        else:
            print('La estrella '+star+' no es apta para analisis de parametros fisicos')
            NoApta=pd.DataFrame({"ID":[star]})
            NoAptas=pd.concat([NoAptas,NoApta],ignore_index=True)
    elif StarType == "RRc":
        print('Es un RRc')
        coef_Fou=salida    
        for column_name in coef_Fou.columns[1:]:
            globals()[column_name] = coef_Fou[column_name].values
        
        
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
        try:
            FeHZWnem=(-b+sqrt(b**2-4*a*c))/(2*a)
        except:
            print("no jala la cuadratica")
            continue
        
        
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
        try:
            M=10**(logM)
        except:
            print("no jala el log")
            continue
        
        
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
        ParFis_RRc=pd.DataFrame({'ID':star,'[Fe/H]_ZWNem':FeHZWnem,'[Fe/H]_Nem':FeH_nem,
                                  '[Fe/H]_ZW':FeHzw,'[Fe/H]_UVES':FeHuves,
                                  'Mv':Mv,'log T_eff':Teff,'log (L/Lo)':L,'M/Mo':M,
                                  'R/Ro':R,'D(pc)':D})
        
        ParFisC=pd.concat([ParFisC,ParFis_RRc],ignore_index=True)
    else:
        print('Estrella de tipo '+StarType+", No se realiza el analisis")
print("Gracias por usar el mamalon, vuelva pronto")    


# Calcular el promedio y la desviación estándar de cada columna en ParFisAB
promedioAB = [nominal_values(ParFisAB[col]).mean() for col in ParFisAB.columns if col != "ID"]
desviacionAB = [nominal_values(ParFisAB[col]).std() for col in ParFisAB.columns if col != "ID"]

# Calcular el promedio y la desviación estándar de las incertidumbres en ParFisAB
promedioIncertidumbreAB = [std_devs(ParFisAB[col]).mean() for col in ParFisAB.columns if col != "ID"]
desviacionIncertidumbreAB = [std_devs(ParFisAB[col]).std() for col in ParFisAB.columns if col != "ID"]

# Calcular lo mismo para ParFisC
promedioC = [nominal_values(ParFisC[col]).mean() for col in ParFisC.columns if col != "ID"]
desviacionC = [nominal_values(ParFisC[col]).std() for col in ParFisC.columns if col != "ID"]

# Calcular el promedio y la desviación estándar de las incertidumbres en ParFisC
promedioIncertidumbreC = [std_devs(ParFisC[col]).mean() for col in ParFisC.columns if col != "ID"]
desviacionIncertidumbreC = [std_devs(ParFisC[col]).std() for col in ParFisC.columns if col != "ID"]

avgAB = [unumpy.uarray(v, i) for v, i in zip(promedioAB, promedioIncertidumbreAB)]
desvAB = [unumpy.uarray(v, i) for v, i in zip(desviacionAB, desviacionIncertidumbreAB)]
avgC = [unumpy.uarray(v, i) for v, i in zip(promedioC, promedioIncertidumbreC)]
desvC = [unumpy.uarray(v, i) for v, i in zip(desviacionC, desviacionIncertidumbreC)]


avgAB=["avg"]+avgAB
avgC=["avg"]+avgC

desvAB=["std_dev"]+desvAB
desvC=["std_dev"]+desvC

ParFisAB.loc[len(ParFisAB)] = avgAB
ParFisAB.loc[len(ParFisAB)] = desvAB

ParFisC.loc[len(ParFisC)] = avgC
ParFisC.loc[len(ParFisC)] = desvC

if guardar== "s":
    ParFisAB.to_csv(f"{ruta_salida}/RRLyrae_AB.txt" , index=False, sep='\t', header=True)
    ParFisAB_noDm.to_csv(f"{ruta_log}/RRLyrae_AB_noDM.txt" , index=False, sep='\t', header=True)
    ParFisC.to_csv(f"{ruta_salida}/RRLyrae_C.txt" , index=False, sep='\t', header=True)
    
    Fou_params.to_csv(f"{ruta_salida}/parametros_fourier.txt" , index=False, sep='\t', header=True) 
    RRL = fst[(fst['Type'] == "RRLyr") ]
    Fou_params_RRlyr = Fou_params[Fou_params["ID"].isin(RRL["ID"])]
    Fou_params_RRlyr.to_csv(f"{ruta_salida}/parametros_fourier_RRLyr.txt" , index=False, sep='\t', header=True)
    Arm_fou.to_csv(f"{ruta_salida}/armonicos_fourier.txt" , index=False, sep='\t', header=True)
    Tabla_General.to_csv(f"{ruta_salida}/Tabla_general.txt", index=False, sep='\t', header=True)

    Dm_out.to_csv(f"{ruta_salida}/Dms.txt" , index=False, sep='\t', header=True)
    NoFiles.to_csv(f"{ruta_log}/NoFiles.txt" , index=False, sep='\t', header=True)
    NoAptas.to_csv(f"{ruta_log}/NoAptasDm.txt" , index=False, sep='\t', header=True)
    if ajuste0 ==True:
        LogAjuste.to_csv(f"{ruta_log}/LogAjuste.txt" , index=False, sep='\t', header=True)
        
    
# ============================ Se hace el diagrama de baily ==================== 

# Se seleccionan solo las RR Lyraes para graficar
P_ab=fst[fst["Subtype"].isin(["RRab","RRabBl"])]
P_c=fst[fst["Subtype"].isin(["RRc","RRcBl","RRd","BL Her"])]

A_vi=A_vi[~A_vi["A_v"].isin([0])]
A_vi=A_vi[~A_vi["A_i"].isin([0])]
    
ab=A_vi.merge(P_ab,on=["ID","Subtype"],how="inner")[["ID","Subtype", "A_v", "A_i", "Max_V", "Min_V", "Max_I", "Min_I","P_1"]]
ab["Plog"]=np.log10(ab["P_1"])
c=A_vi.merge(P_c,on=["ID","Subtype"],how="inner")[["ID","Subtype", "A_v", "A_i", "Max_V", "Min_V", "Max_I", "Min_I","P_1"]]
c["Plog"]=np.log10(c["P_1"])

# Se guarda el DF con el que se hace el diagrama de Bayli
Baylie= pd.concat([ab,c],axis=0)[["ID","Subtype","A_v","A_i","P_1","Plog"]]
if guardar=="s":
    Baylie.to_csv(f"{ruta_salida}/Bailey.txt",index=False,sep="\t",header=True)

ids_rrab = fst[fst["Subtype"] == "RRabBl"]["ID"].unique()
ab_bl = ab[ab["ID"].isin(ids_rrab)]
ab_nobl = ab[~ab["ID"].isin(ids_rrab)]


ids_rrc = fst[fst["Subtype"] == "RRcBl"]["ID"].unique()
c_bl = c[c["ID"].isin(ids_rrc)]
c_nobl = c[~c["ID"].isin(ids_rrc)]


P_ab_t = np.linspace(0.4, 0.8, 50)
P_c_t = np.linspace(0.2, 0.5, 50)
logP_ab_t = np.log10(P_ab_t)
logP_c_t = np.log10(P_c_t)


Av_ab_t = -2.627 - 22.046 * logP_ab_t - 30.876 * logP_ab_t**2  #corroborada por el beto
Av_ab_tp = -2.627 - 22.046 * (-.06 + logP_ab_t) - 30.876 * (-.06 + logP_ab_t)**2  #corroborado por el betesis 
Av_c_t1 = -3.95 + 30.17 * P_c_t - 51.35 * P_c_t**2   #corrobrada por el beturro
Av_c_t2 = -9.75 + 57.3 * P_c_t - 80 * P_c_t**2   # corroborado por el betornado 

Ai_c = -2.72 + 20.78 * P_c_t - 35.51 * P_c_t**2 # Corroborada por el betocino
Ai_c1 = (-11) * logP_c_t**2 - (9.1) * logP_c_t - (1.56) # corroborada por el betostada
Ai_c2 = (-11-4.7) * logP_c_t**2 - (9.1-4.18) * logP_c_t - (1.56-.92) # esta no se usa por razones
Ai_ab1 = -1.64 - 13.78 * logP_ab_t - 19.30 * logP_ab_t**2 # corroborada por el beto
Ai_ab2 = -0.89 - 11.46 * logP_ab_t - 19.30 * logP_ab_t**2

max_y0 = max(Av_ab_t.max(), Av_ab_tp.max(), Av_c_t1.max(), Av_c_t2.max()) + 0.2
max_y1 = max(Ai_c.max(),Ai_c1.max(), Ai_c2.max(), Ai_ab1.max(), Ai_ab2.max()) + .2

# Grafica de Bailey
fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

#Av
# Curvas
axs[0].plot(logP_ab_t, Av_ab_t, color='black', label="Av_ab_t")
axs[0].plot(logP_ab_t, Av_ab_tp, linestyle="--", color='black', label="Av_ab_tp")
axs[0].plot(logP_c_t, Av_c_t1, color='orange', label="Av_c_t1")
axs[0].plot(logP_c_t, Av_c_t2,linestyle="--", color='orange', label="Av_c_t2")

# Estrellas
axs[0].plot(ab_nobl["Plog"], ab_nobl["A_v"], ".", color="b", linestyle='None',markersize=10)
axs[0].plot(c_nobl["Plog"], c_nobl["A_v"], ".", color="green", linestyle='None',markersize=10)
axs[0].plot(ab_bl["Plog"], ab_bl["A_v"], "v", markerfacecolor="b", markeredgecolor="b", linestyle="none")
axs[0].plot(c_bl["Plog"], c_bl["A_v"], "v", markerfacecolor="green", markeredgecolor="green", linestyle="none")

axs[0].set_ylabel(r"$A_V$")
axs[0].set_title("M 22")
axs[0].set_ylim(0, max_y0)

# Ai
# Curvas
axs[1].plot(logP_c_t, Ai_c, color='orange', label="Ai_c")
axs[1].plot(logP_c_t, Ai_c1, linestyle="--", color='orange', label="Ai_c1")
axs[1].plot(logP_ab_t, Ai_ab1, color='black', label="Ai_ab1")
axs[1].plot(logP_ab_t, Ai_ab2, linestyle="--", color='black', label="Ai_ab2", linewidth=2)

# Estrellas
axs[1].plot(ab_nobl["Plog"], ab_nobl["A_i"],".", color="blue",linestyle="none",markersize=10)
axs[1].plot(c_nobl["Plog"], c_nobl["A_i"], ".", color="green", linestyle='None',markersize=10)
axs[1].plot(ab_bl["Plog"], ab_bl["A_i"], "v", markerfacecolor="blue", markeredgecolor="blue", linestyle="none")
axs[1].plot(c_bl["Plog"], c_bl["A_i"], "v", markerfacecolor="green", markeredgecolor="green", linestyle="none")

axs[1].set_ylabel(r"$A_I$")
axs[1].set_xlabel(r"$\log P$")
axs[1].set_ylim(0, max_y1)

texts_av = []
texts_ai = []


def simplify_id(id_str):
    return id_str.split("-")[-1] if id_str.startswith("OGLE-") else id_str


for df in [ab_nobl, ab_bl, c_nobl, c_bl]:
    for _, row in df.iterrows():
        label = simplify_id(row["ID"])
        texts_av.append(axs[0].text(row["Plog"], row["A_v"], label, fontsize=6))

for df in [ab_nobl, ab_bl, c_nobl, c_bl]:
    for _, row in df.iterrows():
        label = simplify_id(row["ID"])
        texts_ai.append(axs[1].text(row["Plog"], row["A_i"], label, fontsize=6))


adjust_text(texts_av, ax=axs[0],expand_text=(1.3, 1.5), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
adjust_text(texts_ai, ax=axs[1],expand_text=(1.3, 1.5), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))


for ax in axs:
    ax.set_xlim(-0.7, 0)
    ax.minorticks_on() 
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
    ax.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
    ax.grid(False)

plt.tight_layout()

if guardar =="s":
    plt.savefig(f"{ruta_graficos}/Bailey.pdf", bbox_inches='tight')
plt.show()
    
    
    