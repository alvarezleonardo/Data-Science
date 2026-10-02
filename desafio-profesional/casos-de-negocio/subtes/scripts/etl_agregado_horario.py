#!/usr/bin/env python3
"""
ETL del caso Subtes: de los CSV crudos de molinetes al agregado horario.

Entrada : datasets/historico_2014.csv .. historico_2026.csv
          13 archivos, ~14 GB, 139.928.997 filas, en seis formatos distintos.
Salida  : datasets/agregado-horario.csv
          fecha,anio,mes,dia_semana,hora,linea,estacion,pax
          ~6,74 millones de filas, agregadas por fecha + hora + linea + estacion.

Por que no usa pandas
---------------------
El volumen de entrada (14 GB) no entra en memoria, y la lectura tiene que ser por
streaming de todos modos. Con la libreria estandar el proceso no depende de que el
entorno tenga pandas instalado, lo cual importa porque este script es el que produce
el dataset que despues si se analiza con pandas.

Las tres cosas delicadas que resuelve
-------------------------------------
1. Seis formatos de archivo distintos entre los 13 anios (separador, orden y nombre
   de columnas, comillas envolviendo la linea entera en 2022-2026).
2. Fechas corrompidas en 2020, 2021 y agosto de 2025. Ver documentacion/problema-fechas.md:
   las fechas ambiguas (ambos campos <= 12) quedaron invertidas y las no ambiguas no.
   Se corrigen en dos pasadas, deduciendo el mes de cada bloque contiguo.
3. Nombres de estacion con encoding roto de forma distinta en cada anio, mas 10 alias.
   Ver documentacion/diccionario-estaciones.md.

Validado contra la suma directa de la columna de pasajeros de los 13 archivos: coincide
exactamente, anio por anio.

Uso:
    python3 etl_agregado_horario.py              # los 13 anios
    python3 etl_agregado_horario.py 2020 2021    # solo algunos
"""

import collections
import datetime
import json
import os
import sys
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.environ.get("SUBTES_DATASETS", os.path.join(AQUI, "..", "datasets"))
PARCIALES = os.environ.get("SUBTES_PARCIALES", os.path.join(DATASETS, "_parciales"))

# Posicion de cada columna por anio, 1-based: (separador, fecha, desde, linea, estacion, total)
FORMATOS = {
    2014: (",", 1, 2, 4, 7, 11),
    2015: (",", 2, 3, 5, 7, 11),
    2016: (",", 1, 2, 4, 7, 11),
    2017: (",", 2, 3, 5, 8, 12),   # trae una columna indice V1 al principio
    2018: (",", 1, 2, 4, 6, 10),
    2019: (",", 2, 3, 5, 7, 11),
    2020: (",", 1, 2, 4, 6, 10),
    2021: (";", 2, 3, 5, 7, 11),
}
for _anio in range(2022, 2027):
    FORMATOS[_anio] = (";", 2, 3, 5, 7, 11)

# Alias verificados cruzando el padron de estaciones contra historico_2018.csv
ALIAS = {
    "FLORES": "SAN JOSE DE FLORES",
    "LEANDRO N. ALEM": "LEANDRO ALEM",
    "ROSAS": "JUAN MANUEL DE ROSAS",
    "GENERAL SAN MARTIN": "SAN MARTIN",
    "MARIANO MORENO": "MORENO",
    "MINISTRO CARRANZA": "CARRANZA",
    "GENERAL BELGRANO": "BELGRANO",
    "HUMBERTO I": "HUMBERTO 1°",
    "PATRICIOS": "PARQUE PATRICIOS",
    "PZA. DE LOS VIRREYES": "PLAZA DE LOS VIRREYES",
}

DIAS_DEL_MES = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 29 en feb: se valida con date()
LINEAS_VALIDAS = ("A", "B", "C", "D", "E", "H")


def limpiar_linea(texto):
    """Saca el salto final y las comillas que envuelven la linea completa (2022-2026)."""
    texto = texto.rstrip("\r\n")
    if len(texto) > 1 and texto[0] == '"' and texto[-1] == '"':
        texto = texto[1:-1]
    return texto


def clasificar_fecha(cruda, anio):
    """
    Devuelve (tipo, mes):
      tipo  1 -> no ambigua, y 'mes' es su mes
      tipo  0 -> ambigua (ambos campos <= 12): el mes lo define el bloque
      tipo -1 -> ilegible
      tipo -2 -> el anio no coincide con el del archivo
    """
    cruda = cruda.replace('"', "").strip()
    try:
        if "-" in cruda:                       # ISO: YYYY-A-B
            partes = cruda.split("-")
            if len(partes) != 3:
                return (-1, 0)
            anio_leido, a, b = int(partes[0]), int(partes[1]), int(partes[2])
        elif "/" in cruda:                     # A/B/YYYY
            partes = cruda.split("/")
            if len(partes) != 3:
                return (-1, 0)
            a, b, anio_leido = int(partes[0]), int(partes[1]), int(partes[2])
        else:
            return (-1, 0)
    except ValueError:
        return (-1, 0)

    if anio_leido != anio:
        return (-2, 0)
    if a > 12 and b > 12:
        return (-1, 0)
    if b > 12:
        return (1, a)                          # b es el dia, a el mes
    if a > 12:
        return (1, b)                          # a es el dia, b el mes
    return (0, 0)


def campos_fecha(cruda):
    """Los dos campos variables de la fecha, en el orden en que vienen."""
    cruda = cruda.replace('"', "").strip()
    if "-" in cruda:
        partes = cruda.split("-")
        return (int(partes[1]), int(partes[2]))
    partes = cruda.split("/")
    return (int(partes[0]), int(partes[1]))


def resolver_fecha(cruda, anio, mes_bloque):
    """
    Con el mes del bloque ya conocido, devuelve (fecha ISO, dia de semana) o None.
    Uno de los dos campos es el mes; el otro es el dia.
    """
    a, b = campos_fecha(cruda)
    mes = mes_bloque
    if a == mes and b == mes:
        dia = mes
    elif a == mes:
        dia = b
    elif b == mes:
        dia = a
    elif a == b:
        mes = dia = a                          # 01/01 dentro de otro bloque: no depende del bloque
    else:
        return None
    if not (1 <= mes <= 12 and 1 <= dia <= DIAS_DEL_MES[mes]):
        return None
    try:
        fecha = datetime.date(anio, mes, dia)
    except ValueError:                          # 29 de febrero en anio no bisiesto
        return None
    return fecha.isoformat(), fecha.weekday()


def normalizar_linea(cruda):
    """('ok', letra) o (motivo_de_descarte, None)."""
    texto = cruda.replace('"', "").strip().upper()
    if texto == "":
        return ("vacio", None)
    if "PRUEBA" in texto:
        return ("prueba", None)
    if texto.startswith("LINEA"):
        texto = texto[5:]
    texto = texto.lstrip("_").strip()
    if texto == "PM":
        return ("premetro", None)              # el Premetro aparece solo desde 2025-06
    if texto in LINEAS_VALIDAS:
        return ("ok", texto)
    return ("linea_invalida", None)


def normalizar_estacion(cruda):
    """('ok', nombre canonico) o (motivo_de_descarte, None)."""
    texto = cruda.replace('"', "").strip().upper()
    if texto == "":
        return ("vacio", None)
    if "PRUEBA" in texto:
        return ("prueba", None)
    # Los dos unicos nombres con caracteres no ASCII llegan corrompidos distinto en cada anio
    if texto.startswith("AG") and texto.endswith("ERO") and 5 <= len(texto) <= 14:
        texto = "AGÜERO"
    elif texto.startswith("SAENZ PE"):
        texto = "SAENZ PEÑA"
    for sufijo in (".B", ".D", ".H"):          # sufijo de linea, redundante con la columna
        if texto.endswith(sufijo):
            texto = texto[:-2]
            break
    texto = ALIAS.get(texto, texto)
    if "TALLER" in texto and "BONIFACIO" in texto:
        return ("taller", None)                # es un taller, no una estacion de pasajeros
    return ("ok", texto)


def detectar_bloques(ruta, anio, sep, col_fecha):
    """
    Pasada 1: devuelve [(fila_inicio, mes)] de cada bloque contiguo.

    Los archivos son bloques de un mes por grupo de lineas, concatenados. Dentro de un
    bloque el mes es constante, y como todo mes tiene dias > 12, todo bloque contiene
    fechas no ambiguas que revelan su mes.

    La frontera entre dos bloques no se pone en la primera fila no ambigua del segundo,
    porque las ultimas filas del primero pueden ser ambiguas: se corre hasta despues de
    la ultima fila ambigua que solo sea compatible con el mes anterior.
    """
    cache = {}
    bloques = []
    mes_actual = None
    fila_ultima_no_amb = -1
    ambiguas_pendientes = {}
    conflictos = []

    with open(ruta, encoding="latin-1", newline="") as f:
        f.readline()
        for i, linea in enumerate(f):
            partes = limpiar_linea(linea).split(sep, col_fecha + 1)
            if len(partes) <= col_fecha:
                continue
            cruda = partes[col_fecha]
            if cruda not in cache:
                cache[cruda] = clasificar_fecha(cruda, anio)
            tipo, mes = cache[cruda]

            if tipo == 1:
                if mes != mes_actual:
                    if mes_actual is None:
                        inicio = 0
                    else:
                        inicio = fila_ultima_no_amb + 1
                        ultima_del_previo = -1
                        primera_del_nuevo = float("inf")
                        for amb, (desde, hasta) in ambiguas_pendientes.items():
                            cs = campos_fecha(amb)
                            compat_previo = mes_actual in cs
                            compat_nuevo = mes in cs
                            if compat_previo and not compat_nuevo and hasta > ultima_del_previo:
                                ultima_del_previo = hasta
                            if compat_nuevo and not compat_previo and desde < primera_del_nuevo:
                                primera_del_nuevo = desde
                        if ultima_del_previo >= 0:
                            inicio = ultima_del_previo + 1
                        if primera_del_nuevo < ultima_del_previo:
                            conflictos.append((mes_actual, mes, primera_del_nuevo, ultima_del_previo))
                    bloques.append((inicio, mes))
                    mes_actual = mes
                ambiguas_pendientes = {}
                fila_ultima_no_amb = i
            elif tipo == 0:
                rango = ambiguas_pendientes.get(cruda)
                if rango is None:
                    ambiguas_pendientes[cruda] = [i, i]
                else:
                    rango[1] = i

    return bloques, conflictos


def procesar(anio):
    """Procesa un anio completo y deja su parcial y sus estadisticas en PARCIALES."""
    sep, col_fecha, col_desde, col_linea, col_est, col_total = FORMATOS[anio]
    col_fecha -= 1; col_desde -= 1; col_linea -= 1; col_est -= 1; col_total -= 1
    ruta = os.path.join(DATASETS, "historico_%d.csv" % anio)
    ultima_col = max(col_fecha, col_desde, col_linea, col_est, col_total)

    bloques, conflictos = detectar_bloques(ruta, anio, sep, col_fecha)

    agregado = collections.defaultdict(int)
    descartes = collections.Counter()
    descartes_pax = collections.Counter()
    cache_fecha, cache_linea, cache_est, cache_hora, cache_mol = {}, {}, {}, {}, {}
    suma_cruda = filas_crudas = conservadas = pax_conservado = 0

    indice_bloque = 0
    mes_bloque = bloques[0][1] if bloques else None
    proximo_inicio = bloques[1][0] if len(bloques) > 1 else float("inf")

    with open(ruta, encoding="latin-1", newline="") as f:
        f.readline()
        for i, linea_cruda in enumerate(f):
            while i >= proximo_inicio:
                indice_bloque += 1
                mes_bloque = bloques[indice_bloque][1]
                proximo_inicio = (bloques[indice_bloque + 1][0]
                                  if indice_bloque + 1 < len(bloques) else float("inf"))

            partes = limpiar_linea(linea_cruda).split(sep)
            if len(partes) <= ultima_col:
                descartes["malformada"] += 1
                continue

            bruto = partes[col_total].replace('"', "").strip()
            if bruto in ("", "NA"):
                pax = 0
            else:
                try:
                    pax = int(float(bruto))    # 2015, 2018 y 2019 lo traen como decimal
                except ValueError:
                    descartes["total_ilegible"] += 1
                    continue
            suma_cruda += pax
            filas_crudas += 1

            cruda_linea = partes[col_linea]
            if cruda_linea not in cache_linea:
                cache_linea[cruda_linea] = normalizar_linea(cruda_linea)
            estado_linea, letra = cache_linea[cruda_linea]

            cruda_est = partes[col_est]
            if cruda_est not in cache_est:
                cache_est[cruda_est] = normalizar_estacion(cruda_est)
            estado_est, estacion = cache_est[cruda_est]

            motivo = None
            if estado_linea == "vacio" or estado_est == "vacio":
                motivo = "vacio"
            elif estado_linea == "prueba" or estado_est == "prueba":
                motivo = "prueba"
            else:
                molinete = partes[col_linea + 1] if col_linea + 1 < len(partes) else ""
                if molinete not in cache_mol:
                    cache_mol[molinete] = "prueba" in molinete.lower()
                if cache_mol[molinete]:
                    motivo = "prueba"
                elif estado_linea == "premetro":
                    motivo = "premetro"
                elif estado_linea != "ok":
                    motivo = estado_linea
                elif estado_est == "taller":
                    motivo = "taller"

            fecha = None
            if motivo is None:
                cruda_fecha = partes[col_fecha]
                clave = (cruda_fecha, mes_bloque)
                if clave not in cache_fecha:
                    tipo, mes = clasificar_fecha(cruda_fecha, anio)
                    if tipo == -2:
                        cache_fecha[clave] = -2
                    elif tipo == -1:
                        cache_fecha[clave] = None
                    elif tipo == 1:
                        cache_fecha[clave] = resolver_fecha(cruda_fecha, anio, mes)
                    else:
                        cache_fecha[clave] = (resolver_fecha(cruda_fecha, anio, mes_bloque)
                                              if mes_bloque else None)
                fecha = cache_fecha[clave]
                if fecha == -2:
                    motivo = "anio_distinto"
                elif fecha is None:
                    motivo = "fecha_no_corregible"

            if motivo:
                descartes[motivo] += 1
                descartes_pax[motivo] += pax
                continue

            desde = partes[col_desde].replace('"', "").strip()
            if desde not in cache_hora:
                cache_hora[desde] = int(desde.split(":")[0])
            hora = cache_hora[desde]

            agregado[(fecha[0], fecha[1], hora, letra, estacion)] += pax
            conservadas += 1
            pax_conservado += pax

    os.makedirs(PARCIALES, exist_ok=True)
    salida = os.path.join(PARCIALES, "part_%d.csv" % anio)
    with open(salida, "w", encoding="utf-8") as o:
        for clave in sorted(agregado, key=lambda k: (k[0], k[2], k[3], k[4])):
            fecha_iso, dia_semana, hora, letra, estacion = clave
            o.write("%s,%s,%d,%d,%d,%s,%s,%d\n" % (
                fecha_iso, fecha_iso[:4], int(fecha_iso[5:7]), dia_semana,
                hora, letra, estacion, agregado[clave]))

    with open(os.path.join(PARCIALES, "stats_%d.json" % anio), "w") as o:
        json.dump(dict(anio=anio, filas_crudas=filas_crudas, suma_cruda=suma_cruda,
                       bloques=bloques, conflictos=conflictos, conservadas=conservadas,
                       pax_conservado=pax_conservado, filas_salida=len(agregado),
                       descartes=dict(descartes), descartes_pax=dict(descartes_pax)), o)
    return anio


def unir(anios):
    """Une los parciales en el agregado final, ordenado por fecha."""
    final = os.path.join(DATASETS, "agregado-horario.csv")
    with open(final, "w", encoding="utf-8") as o:
        o.write("fecha,anio,mes,dia_semana,hora,linea,estacion,pax\n")
        for anio in sorted(anios):
            parcial = os.path.join(PARCIALES, "part_%d.csv" % anio)
            with open(parcial, encoding="utf-8") as f:
                for linea in f:
                    o.write(linea)
    return final


if __name__ == "__main__":
    anios = [int(a) for a in sys.argv[1:]] or sorted(FORMATOS)
    with Pool(min(10, len(anios))) as pool:
        for anio in pool.imap_unordered(procesar, anios):
            print("listo", anio, flush=True)
    print("agregado en:", unir(anios))
