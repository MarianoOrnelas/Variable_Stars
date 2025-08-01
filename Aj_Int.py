#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:46:36 2025

@author: alberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 11:32:33 2025

@author: alberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.optimize import minimize_scalar

# --- CONSTANTES GLOBALES ---
n_data = 0
carpeta = "" #Es la carpeta donde esta el programa

# --- DEFINICION DE FUNCIONES ---

#i) FUNCION PARA GRAFICAR LA CURVA FASEADA
def plot_fas(fases, star, filtro, guardar):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_data))
    
    for k in range(1, n_data + 1):
        subset = fases[fases["ver"] == k]
        ax.plot(subset["fase"], subset["mag"], ".", color=colors[k-1], markersize=5, label=f'Grupo {k}')
    
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 0.5, 1])
    ax.invert_yaxis()
    ax.set_ylabel(filtro, fontstyle="italic", fontsize=14)
    ax.set_xlabel(r"$\phi$", fontsize=14)
    plt.title(star.split('-')[-1] if star.startswith('OGLE-') else star)
    plt.legend()
    plt.minorticks_on()
    ax.tick_params(
        which='both',      
        direction='in',    
        top=True,          
        right=True,        
        labeltop=False,    
        labelright=False   
    )

    #Guarda la imagen si se da esa opcion al inicio del programa
    if guardar == "s":
        plt.savefig(f"{carpeta}/Fourier/{star}_{filtro}.pdf", bbox_inches='tight')
    plt.show()

#ii) FUNCION PARA EL AJUSTE DE FOURIER

def fourier_fun(n_terms, P, E0):
    def m(x, *params):
        A0 = params[0]
        result = A0
        for n in range(1, n_terms + 1):
            An = params[n]
            Bn = params[n + n_terms]
            result += An * np.cos(2 * np.pi * n * (x - E0) / P) + Bn * np.sin(2 * np.pi * n * (x - E0) / P)
        return result
    return m

# FUNCIÓN FASEAR
def fasear(star, ruta, P, E0, n_terms, prin_group, guardar):
    try:
        datos = pd.read_csv(ruta, sep=r'\s+', header=None)
        # Renombrar columnas
        datos.columns = ['HJD', 'mag', 'err', 'ver'] if len(datos.columns) == 4 else ['HJD', 'mag', 'ver']

            
        faseo = []
        for _, fila in datos.iterrows():
            row_data = {
                'HJD': fila['HJD'],
                'fase': (fila['HJD'] - E0) / P % 1,
                'mag': fila['mag'],
                'ver': fila['ver']
            }
            faseo.append(row_data)
            
            if 0.7 <= row_data['fase'] <= 1.0:
                faseo.append({**row_data, 'fase': row_data['fase'] - 1})
            if 0.0 <= row_data['fase'] <= 0.3:
                faseo.append({**row_data, 'fase': row_data['fase'] + 1})
                
        return pd.DataFrame(faseo).sort_values('fase'), datos
    
    except Exception as e:
        print(f"Error en fasear: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()
    
# FUNCIONES PARA EL PERIODO
def metodo_de_la_cuerda(periodo, tiempos, valores):
    fases = (tiempos % periodo) / periodo
    orden = np.argsort(fases)
    #fases_ord = fases[orden]
    valores_ord = valores[orden]
    #dx = np.diff(fases_ord)
    dy = np.diff(valores_ord)
    #longitud_total = np.sum(np.sqrt(dx**2 + dy**2))
    longitud_total = np.sum(dy)
    return longitud_total



def buscar_periodo_optimo(rutaI,rutaV,E_max, P ,a= None,b=None, n_valores = 10000, m_iteraciones = 5, 
                          delta = 0.01, precision_final = 1e-8):
    if a is None:
        a = P - 0.02
    if b is None:
        b = P + 0.02
        
    datosI = None
    datosV = None
    
    if os.path.exists(rutaI):
        datosI = np.loadtxt(rutaI)
    else:
        print(f"[Advertencia] No se encontró el archivo: {rutaI}")
    
    if os.path.exists(rutaV):
        datosV = np.loadtxt(rutaV)
    else:
        print(f"[Advertencia] No se encontró el archivo: {rutaV}")

    
    # Seleccionar los datos válidos
    if datosI is not None and (datosV is None or len(datosI) > len(datosV)):
        t = datosI[:, 0]
        m = datosI[:, 1]
    elif datosV is not None:
        t = datosV[:, 0]
        m = datosV[:, 1]
    else:
        print("[Error] No se encontró ninguno de los archivos necesarios (I o V).")
        # Aquí puedes salir del programa o lanzar una excepción, según prefieras
        raise FileNotFoundError("Ambos archivos de datos están ausentes.")
    # Paso 1: Refinamiento con cuadrícula (MLC)
    for m_iter in range(m_iteraciones):
        delta_P = np.linspace(a, b, n_valores)
        mejores = []
        for P in delta_P:
            if P == 0.:
                continue
            fases = (t - E_max) / P
            fases = fases - np.floor(fases)
            orden = np.argsort(fases)
            m_ordenado = m[orden]
            longitud = np.abs(np.diff(m_ordenado)).sum()
            mejores.append((P, longitud))
        mejores.sort(key = lambda x: x[1])
        periodo_aprox = mejores[0][0]
        a = periodo_aprox - delta
        b = periodo_aprox + delta
        delta /= 10  # Refina más

    # Paso 2: Optimización precisa con minimize_scalar
    def funcion_objetivo(p):
        return metodo_de_la_cuerda(p, t, m)

    res = minimize_scalar(funcion_objetivo, bounds = (a, b), method = 'bounded', 
                          options = {'xatol': precision_final})
    periodo_final = res.x
    print(f"\n Periodo optimizado: {periodo_final:.10f}")

    return periodo_final

# --- PROGRAMA PRINCIPAL ---
def main():
    global n_data, carpeta
    
    # Configuración inicial
    ruta = Path(__file__).parent.resolve()
    carpeta = str(ruta)

    # Entrada de parametros por el usuario
    print("=== AJUSTE DE CURVAS DE LUZ ===")
    star = input("ID de la estrella (ej: V1 u OGLE-BLG-RRLYR-0001): ").strip()
    filter =input("¿En que filtro trabajaras?: ").strip()
    n_data = int(input("¿Cuántos grupos de datos se están usando? "))
    prin_group = int(input("¿Cuál es el grupo principal de datos? "))
    while True:
        guardar = input("¿Guardar gráficos? (s/n): ").strip().lower()
        if guardar in ['s', 'n']:
            break
        print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
    
    # Crear estructura de carpetas
    Path(f"{carpeta}/Data_fit/{filter}/").mkdir(parents=True, exist_ok=True)
    Path(f"{carpeta}/Data_fas/{filter}/").mkdir(parents=True, exist_ok=True)
    Path(f"{carpeta}/graficos/fourier/").mkdir(parents=True, exist_ok=True)
    
    # Definicion de rutas
    rutaI=f"{carpeta}/DATA_RAW/I/{star}.dat"
    rutaV=f"{carpeta}/DATA_RAW/V/{star}.dat"
    ruta_v = f"{carpeta}/DATA_RAW/{filter}/{star}.dat"
    ruta_ajust = f"{carpeta}/Data_fit/{filter}/{star}.dat"
    ruta_faseado = f"{carpeta}/Data_fas/{filter}/{star}.fas"
    
    # Leer parámetros
    fst = pd.read_csv(f"{carpeta}/tablas/fst.txt", sep=r'\s+')
    params = fst[fst['ID'] == star].iloc[0]
    P, E0, n_terms = params['P_1'], params['T0_1'], params['Arms']

    # --- PASO 1: Corrección del punto cero ---
    print("\n--- PASO 1: Corrección del punto cero ---")
    datos_ajustados =pd.read_csv(ruta_v,  sep=r'[,\s]+', header=None)
    datos_ajustados.columns=['HJD','mag','err','ver']
    fases_v,_= fasear(star, ruta_v, P, E0, n_terms, prin_group, guardar)
    if fases_v.empty:
        return

    correcciones = {g: 0.0 for g in range(1, n_data + 1)}

    for grupo in range(1, n_data + 1):
        if grupo == prin_group:
            continue
            
        while True:
            plot_fas(fases_v, star, filter, guardar)
            # Validación de respuesta
            while True:
                resp = input(f"¿Ajuste correcto para grupo {grupo}? (s/n): ").lower().strip()
                if resp in ['s', 'n']:
                    break
                print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
            if resp == 's':
                break
                
            correccion = float(input("Corrección a aplicar (ej: +0.1): "))
            correcciones[grupo] += correccion
            
            # Aplicar corrección a ambos DataFrames
            mask_grupo = datos_ajustados['ver'] == grupo
            datos_ajustados.loc[mask_grupo, 'mag'] += correccion
            
            
            mask_fases = fases_v['ver'] == grupo
            fases_v.loc[mask_fases, 'mag'] += correccion

        print(f"Corrección total grupo {grupo}: {correcciones[grupo]:.3f}")

    # Guardar datos ajustados
    datos_ajustados['HJD'] = datos_ajustados['HJD'].map('{:.6f}'.format)
    datos_ajustados['mag'] = datos_ajustados['mag'].map('{:.4f}'.format)
    datos_ajustados['err'] = datos_ajustados['err'].map('{:.2f}'.format)
    datos_ajustados.to_csv(ruta_ajust, sep='\t',header=False, index=False)
    print(f"\nDatos ajustados guardados en: {ruta_ajust}")
    
    # --- PASO 2: Ajuste de periodo ---
    print("\n--- PASO 2: Ajuste de periodo ---")
    
    while True:
        # Calcular fases y graficar con el periodo actual
        fases_v, _ = fasear(star, ruta_ajust, P, E0, n_terms, prin_group, guardar)
        plot_fas(fases_v, star, filter, guardar)
    
        # Confirmar si el periodo actual es correcto
        while True:
            resp = input(f"Periodo actual: {P:.6f}. ¿Es correcto? (s/n): ").lower().strip()
            if resp in ['s', 'n']:
                break
            print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
        
        if resp == 's':
            break  # Salir del ciclo si el periodo es aceptado
    
        # Preguntar si usar el buscador automático
        while True:
            usar_auto = input("¿Quieres usar el buscador automático para encontrar un nuevo periodo? (s/n): ").lower().strip()
            if usar_auto in ['s', 'n']:
                break
            print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
        
        if usar_auto == 's':
            # Buscar sin especificar a y b (usar valores por defecto internos)
            nuevo_P = buscar_periodo_optimo(rutaI, rutaV, E0, P)
            print(f"Periodo sugerido automáticamente: {nuevo_P:.6f}")
            P = nuevo_P
    
            # Graficar para este nuevo periodo
            fases_v, _ = fasear(star, ruta_ajust, P, E0, n_terms, prin_group, guardar)
            plot_fas(fases_v, star, filter, guardar)
    
            # Confirmar si este nuevo periodo es correcto
            while True:
                resp = input(f"¿Aceptar este nuevo periodo? (s/n): ").lower().strip()
                if resp in ['s', 'n']:
                    break
                print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
            
            if resp == 's':
                break  # Periodo aceptado
    
            # Si no fue aceptado, preguntar si se quiere usar intervalo personalizado
            while True:
                usar_ab = input("¿Quieres especificar un intervalo [a, b] para buscar el periodo? (s/n): ").lower().strip()
                if usar_ab in ['s', 'n']:
                    break
                print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
            
            if usar_ab == 's':
                a = float(input("Ingresa el valor de 'a' (inicio del intervalo): "))
                b = float(input("Ingresa el valor de 'b' (fin del intervalo): "))
                nuevo_P = buscar_periodo_optimo(rutaI, rutaV, E0, P, a=a, b=b)
                print(f"Periodo sugerido con intervalo [{a}, {b}]: {nuevo_P:.6f}")
                P = nuevo_P
    
        else:
            # Ingreso manual de periodo
            P = float(input("Ingresa manualmente el nuevo periodo (días): "))
    
        # Graficar después de cualquier cambio de periodo
        fases_v, _ = fasear(star, ruta_ajust, P, E0, n_terms, prin_group, guardar)
        plot_fas(fases_v, star, filter, guardar)
    
        # Confirmar si el nuevo periodo (manual o con a/b) es correcto
        while True:
            resp = input(f"¿Aceptar este nuevo periodo: {P:.6f}? (s/n): ").lower().strip()
            if resp in ['s', 'n']:
                break
            print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
        
        if resp == 's':
            break  # Periodo final confirmado

    # --- PASO 3 CON VALIDACIÓN DE ENTRADA ---
    print("\n--- PASO 3: Época de máximo/mínimo ---")
    print(f"Época inicial leída del archivo: {E0}")
    
  
    # Primera verificación con E0 original
    fases_v, _ = fasear(star, ruta_ajust, P, E0, n_terms, prin_group, guardar)
    plot_fas(fases_v, star, filter, guardar)
    while True:
        resp = input(f"¿Es correcta esta época ({E0})? (s/n): ").lower().strip()
        if resp in ['s', 'n']:
            break
        print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
    if resp == 'n':
        while True:
            resp = input("¿Desea ingresar manualmente una época? (s/n):si su respuesta es no, se buscará automáticamente. ").lower().strip()
            if resp in ['s', 'n']:
                break
            print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
        if resp == 's':
            E0 = float(input("Ingrese manualmente la época (HJD): "))
            fases_v, _ = fasear(star, ruta_ajust, P, E0, n_terms, prin_group, guardar)
            plot_fas(fases_v, star, filter, guardar)
        else:
            # Validación robusta de entrada
            while True:
                tipo = input("¿Buscar máximo o mínimo? (max/min): ").lower().strip()
                if tipo in ['max', 'min']:
                    break
                print("¡Error! Solo se acepta 'max' o 'min'. Intente nuevamente.")
            
            # Búsqueda automática de alternativas
            if tipo == 'min':
                candidatos = fases_v.nlargest(20, 'mag')['HJD'].unique()
            else:
                candidatos = fases_v.nsmallest(20, 'mag')['HJD'].unique()
            
            for i, candidato in enumerate(candidatos):
                if candidato == E0:  # Saltar el valor inicial ya rechazado
                    continue
                    
                fases_v, _ = fasear(star, ruta_ajust, P, candidato, n_terms, prin_group, guardar)
                plot_fas(fases_v, star, filter, guardar)
                while True:
                    resp = input(f"¿Es correcta esta época alternativa ({candidato})? (s/n): ").lower().strip()
                    if resp in ['s', 'n']:
                        break
                    print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
                    
                if resp == 's':
                    E0 = candidato
                    break

            print("No se encontró una época adecuada automáticamente, se tomará la época original")
            fases_v, _ = fasear(star, ruta_ajust, P, E0, n_terms, prin_group, guardar)
            plot_fas(fases_v, star, filter, guardar)
                
    print(f"Época final seleccionada: {E0}")
    # Guardar datos faseados
    # Primero aplicar redondeo a las columnas
    fases=fases_v
    fases['HJD'] = fases['HJD'].round(6)
    fases[['fase', 'mag']] = fases[['fase', 'mag']].round(4)
    fases['ver'] = fases['ver'].astype(int)
    
    # Luego guardar
    fases.to_csv(ruta_faseado, 
                  sep='\t', 
                  columns=['HJD', 'fase', 'mag', 'ver'],
                  header=False, 
                  index=False,
                  float_format='%.6f')  # Solo para HJD (las otras ya están redondeadas)
    
    print(f"\nDatos faseados guardados en: {ruta_faseado}")
    
    # --- PASO 4 CORREGIDO ---
    print("\n--- PASO 4: Número de armónicos ---")
    while True:
        # 1. Ajuste Fourier
        m = fourier_fun(n_terms, P, E0)
        p0 = [1] * (1 + 2 * n_terms)
        res, _ = curve_fit(m, fases_v['HJD'], fases_v['mag'], p0=p0)
        
        # 2. Generar puntos para la curva teórica
        fase_plot = np.linspace(-0.3, 1.3, 500)  # Fases para graficar
        t_plot = fase_plot * P + E0  # Convertir fases a HJD
        y_plot = m(t_plot, *res)  # Calcular magnitudes del modelo
        
        # 3. Graficar PUNTOS y CURVA en el mismo espacio de fases
        plt.figure(figsize=(6.4, 4.8))
        
        # Puntos observados (color por grupo)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_data))
        for k in range(1, n_data + 1):
            subset = fases_v[fases_v["ver"] == k]
            plt.plot(subset["fase"], subset["mag"], ".", color=colors[k-1], markersize=5)
        
        # Curva de Fourier (en negro)
        plt.plot(fase_plot, y_plot, 'k-', linewidth=1.5)
        
        # Formateo del gráfico
        plt.xlim(-0.35, 1.35)
        plt.xticks([0, 0.5, 1])
        plt.gca().invert_yaxis()
        plt.ylabel(filter, fontstyle="italic", fontsize=14)
        plt.xlabel(r"$\phi$", fontsize=14)
        plt.title(f"{star}")
        plt.minorticks_on()
        plt.tick_params(
            which='both',      
            direction='in',    
            top=True,          
            right=True,        
            labeltop=False,    
            labelright=False   
        )
        
        if guardar == "s":
            plt.savefig(f"{carpeta}/Fourier/{star}_ajuste_{n_terms}armonicos.pdf", bbox_inches='tight')
        plt.show()
        
        # 4. Preguntar si el ajuste es satisfactorio
        while True:
            resp = input(f"¿El ajuste con {n_terms} armónicos es correcto? (s/n): ").lower().strip()
            if resp in ['s', 'n']:
                break
            print("¡Error! Solo se acepta 's' o 'n'. Intente nuevamente.")
        if resp == 's':
            break
        n_terms = int(input("Nuevo número de armónicos: "))
    
    # --- RESULTADOS FINALES ---
    print("\n=== RESULTADOS FINALES ===")
    print(f"Correcciones aplicadas por grupo: {correcciones}")
    print(f"Periodo final: {P:.6f} días")
    print(f"Época final: {E0}")
    print(f"Armónicos usados: {n_terms}")
    
    # --- GUARDAR RESULTADOS (CON REEMPLAZO SI EXISTE) ---
    resultados_file = f"{carpeta}/resultados_individuales.txt"
    
    # Crear DataFrame con los nuevos resultados (asegurando tipos correctos)
    nuevos_resultados = pd.DataFrame({
        "ID": [str(star)],  # Asegurar string
        "filtro": [str(filter)],  # Asegurar string
        "correcciones": [str(correcciones)],  # Convertir dict a string
        "periodo": [float(P)],  # Asegurar float
        "epoca": [float(E0)],  # Asegurar float
        "armonicos": [int(n_terms)]  # Asegurar int
    })
    
    # Si el archivo existe, cargarlo y actualizar/reemplazar
    if Path(resultados_file).exists():
        resultados_existentes = pd.read_csv(resultados_file, sep='\t')
        
        # Asegurar tipos de columnas en el DataFrame existente
        resultados_existentes = resultados_existentes.astype({
            'ID': 'str',
            'filtro': 'str',
            'correcciones': 'str',
            'periodo': 'float64',
            'epoca': 'float64',
            'armonicos': 'int64'
        })
        
        # Verificar si ya existe una entrada para esta estrella y filtro
        mask = (resultados_existentes["ID"] == star) & (resultados_existentes["filtro"] == filter)
        
        if mask.any():
            # Reemplazar la fila existente
            resultados_existentes.loc[mask] = nuevos_resultados.iloc[0].values
            resultados_completos = resultados_existentes
        else:
            # Añadir nueva fila
            resultados_completos = pd.concat([resultados_existentes, nuevos_resultados], ignore_index=True)
    else:
        # Crear archivo nuevo
        resultados_completos = nuevos_resultados
    
    # Guardar el archivo (siempre con tabs)
    resultados_completos.to_csv(resultados_file, sep='\t', index=False)
    print(f"\nResultados guardados en: {resultados_file}")

if __name__ == "__main__":
    main()