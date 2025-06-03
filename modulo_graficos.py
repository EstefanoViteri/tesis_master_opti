import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.colors as mcolors

### Funciones para graficar los resultados
def graficar_un_escenario_edades(informacion, variable, escenario, anios):
    colores_tipo = {
    '1_Diesel_12':'#FFB3BA',
    '2_EB_12_360_CD':'#FFDFBA',
    '3_EB_12_450_CD':'#BAFFC9',
    '4_EB_12_250_CD+CT':'#BAE1FF',
    '5_EB_12_320_CD+CT':'#D5BAFF',
    # Agrega más si tienes otros tipos
    }
    df = informacion[escenario][variable]
    tipos = list(df['Tipo de autobus'].unique())
    max_edad_dict = {tipo: df[df['Tipo de autobus'] == tipo]['Edad'].max() for tipo in tipos}
    tipo_edad_dict = {}
    for tipo in tipos:
        edades = sorted(df[df['Tipo de autobus'] == tipo]['Edad'].unique())
        for edad in edades:
            tipo_edad_dict[(tipo, edad)] = []

    for anio in anios:
        df_anio = df[df['Periodo'] == anio]
        count_by = df_anio.groupby(['Tipo de autobus', 'Edad'])['z'].sum()
        for (tipo, edad) in tipo_edad_dict.keys():
            tipo_edad_dict[(tipo, edad)].append(count_by.get((tipo, edad), 0))

    fig, ax = plt.subplots(figsize=(min(35, max(18, 1.4*len(anios))), 11))
    bar_width = 0.7
    bar_positions = np.arange(len(anios))
    bottom = np.zeros(len(anios))
    handles_tipos = {}

    for idx, ((tipo, edad), counts) in enumerate(tipo_edad_dict.items()):
        # Edad 1 es color original, mayor edad más oscuro
        max_edad = max_edad_dict[tipo] if max_edad_dict[tipo] > 1 else 1
        # El degradado: factor 1 para edad=1, 0 para edad=max_edad
        factor = 1 - (edad-1) / (max_edad-1) if max_edad > 1 else 1
        base_rgb = np.array(mcolors.to_rgb(colores_tipo[tipo]))
        # 0.35 = oscuridad máxima, 1 = original (más claro)
        color = base_rgb * (0.35 + 0.65 * factor)
        color = np.clip(color, 0, 1)
        bars = ax.bar(bar_positions, counts, bar_width, bottom=bottom, color=color)
        # Guardar un solo handle para la leyenda por tipo (el más claro)
        if tipo not in handles_tipos and edad == 1:
            handles_tipos[tipo] = bars[0]
        bottom += np.array(counts)
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(count)}',
                        ha='center', va='center', fontsize=10)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(anios, rotation=45, fontsize=14)
    ax.set_xlabel('Años', fontsize=17)
    ax.set_ylabel('Cantidad de autobuses disponibles', fontsize=17)
    plt.yticks(fontsize=14)

    # Leyenda: solo tipos y color original (edad=1)
    labels_tipos = [tipo.replace('_', '-')[2:] for tipo in handles_tipos.keys()]
    ax.legend(handles_tipos.values(), labels_tipos, loc='upper left', bbox_to_anchor=(1, 1),
              frameon=False, fontsize=15, title="Tipo de bus")
    plt.tight_layout(rect=[0, 0.1, 0.85, 1])

    # Leyenda del degradado de edad (ejemplo para el primer tipo)
    from matplotlib.colors import to_rgb
    tipo_demo = tipos[0]  # El primer tipo de tu lista
    max_edad = max_edad_dict[tipo_demo]
    base_rgb = np.array(to_rgb(colores_tipo[tipo_demo]))
    grad = np.linspace(1, 0, max_edad)
    grad_colors = [base_rgb * (0.35 + 0.65 * g) for g in grad]
    grad_img = np.array([grad_colors])
    axins = fig.add_axes([0.15, 0.01, 0.14, 0.045])
    axins.imshow(grad_img, aspect='auto')
    axins.set_xticks(np.linspace(0, max_edad-1, min(max_edad, 5), dtype=int))
    axins.set_xticklabels(np.linspace(1, max_edad, min(max_edad, 5), dtype=int), fontsize=11)
    axins.set_yticks([])
    axins.set_title(f"{tipo_demo.replace('_', '-')[2:]}: Edad", fontsize=12)
    for spine in axins.spines.values():
        spine.set_visible(False)

    plt.show()

def graficar_sensibilidad_variable_v14(modelo, variable_dict, nombre_var: str, eje_tiempo_idx: int = 1):
    """
    Versión 14: Mejora el espacio visual para los símbolos ∞, asegurando separación suficiente del eje.
    """
    data_por_tipo = {}
    rangos = {}

    for idx, var in variable_dict.items():
        try:
            sa_low = var.SAObjLow
            sa_up = var.SAObjUp
            obj_coef = var.Obj
            rangos[idx] = [(sa_low, sa_up) , obj_coef]
            if isinstance(idx, tuple):
                tipo = idx[0]
                tiempo = idx[eje_tiempo_idx]
            else:
                tipo = 'default'
                tiempo = idx
            if tipo not in data_por_tipo:
                data_por_tipo[tipo] = []
            data_por_tipo[tipo].append((tiempo, sa_low, obj_coef, sa_up))
        except AttributeError:
            continue

    n_tipos = len(data_por_tipo)
    if n_tipos == 0:
        print(f"No hay datos de sensibilidad disponibles para '{nombre_var}'. ¿Es un modelo LP?")
        return

    fig, axs = plt.subplots(n_tipos, 1, figsize=(14, 4 * n_tipos), sharex=True)
    if n_tipos == 1:
        axs = [axs]

    bar_width = 0.4
    marker_width = 0.25

    for ax, (tipo, data) in zip(axs, sorted(data_por_tipo.items())):
        data.sort(key=lambda x: x[0])
        tiempo, low, obj, up = zip(*data)

        low_vals = [l for l in low if np.isfinite(l)]
        up_vals = [u for u in up if np.isfinite(u)]
        all_vals = list(obj) + low_vals + up_vals

        y_min_data = min(all_vals) if all_vals else 0
        y_max_data = max(all_vals) if all_vals else 1
        range_span = y_max_data - y_min_data if y_max_data != y_min_data else 1

        # Ampliar más para separar ∞ del tope visual
        y_inf_up = y_max_data + 0.12 * range_span
        y_inf_low = y_min_data - 0.12 * range_span
        y_max = y_inf_up + 0.08 * range_span
        y_min = y_inf_low - 0.08 * range_span

        for t, l, o_, u_ in zip(tiempo, low, obj, up):
            l_draw = y_inf_low if not np.isfinite(l) else l
            u_draw = y_inf_up if not np.isfinite(u_) else u_
            alto = u_draw - l_draw

            ax.add_patch(plt.Rectangle((t - bar_width / 2, l_draw), bar_width, alto,
                                       facecolor='skyblue', alpha=0.4, edgecolor=None, zorder=1))

            if np.isfinite(l):
                ax.plot([t - bar_width / 2, t + bar_width / 2], [l, l], color='red', linewidth=2.5, zorder=2)
            else:
                ax.plot([t - bar_width / 4, t + bar_width / 4], [y_inf_low, y_inf_low],
                        color='red', linewidth=2.5, zorder=2)
                ax.text(t, y_inf_low - 0.015 * range_span, '∞', color='red', fontsize=12,
                        ha='center', va='top')

            if np.isfinite(u_):
                ax.plot([t - bar_width / 2, t + bar_width / 2], [u_, u_], color='red', linewidth=2.5, zorder=2)
            else:
                ax.plot([t - bar_width / 4, t + bar_width / 4], [y_inf_up, y_inf_up],
                        color='red', linewidth=2.5, zorder=2)
                ax.text(t, y_inf_up + 0.015 * range_span, '∞', color='red', fontsize=12,
                        ha='center', va='bottom')

            ax.plot([t - marker_width / 2, t + marker_width / 2], [o_, o_],
                    color='black', linewidth=2.5, zorder=3)

        ax.set_title(f"{nombre_var} - Tipo: {tipo}", fontsize=14)
        ax.set_ylabel("Valor coeficiente", fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.5)

    axs[-1].set_xlabel("Año", fontsize=12)
    plt.tight_layout()
    plt.show()
    return rangos

def graficar_sensibilidad_costo_kwh(dict_info, nombre_var: str, eje_tiempo_idx: int = 1,porcen = False):
    data_por_tipo = {}
    rangos = {}

    for idx, var in dict_info.items():
        try:
            sa_low = var[0][0] if porcen == False else var[0][0] / var[1]
            sa_up = var[0][1] if porcen == False else var[0][1] / var[1]
            obj_coef = var[1] if porcen == False else 1
            if isinstance(idx, tuple):
                tipo = idx[0]
                tiempo = idx[eje_tiempo_idx]
            else:
                tipo = 'default'
                tiempo = idx
            if tipo not in data_por_tipo:
                data_por_tipo[tipo] = []
            data_por_tipo[tipo].append((tiempo, sa_low, obj_coef, sa_up))
        except AttributeError:
            continue

    n_tipos = len(data_por_tipo)
    if n_tipos == 0:
        print(f"No hay datos de sensibilidad disponibles para '{nombre_var}'. ¿Es un modelo LP?")
        return

    fig, axs = plt.subplots(n_tipos, 1, figsize=(14, 4 * n_tipos), sharex=True)
    if n_tipos == 1:
        axs = [axs]

    bar_width = 0.4
    marker_width = 0.25

    for ax, (tipo, data) in zip(axs, sorted(data_por_tipo.items())):
        data.sort(key=lambda x: x[0])
        tiempo, low, obj, up = zip(*data)

        low_vals = [l for l in low if np.isfinite(l)]
        up_vals = [u for u in up if np.isfinite(u)]
        all_vals = list(obj) + low_vals + up_vals

        y_min_data = min(all_vals) if all_vals else 0
        y_max_data = max(all_vals) if all_vals else 1
        range_span = y_max_data - y_min_data if y_max_data != y_min_data else 1

        # Ampliar más para separar ∞ del tope visual
        y_inf_up = y_max_data + 0.12 * range_span
        y_inf_low = y_min_data - 0.12 * range_span
        y_max = y_inf_up + 0.08 * range_span
        y_min = y_inf_low - 0.08 * range_span

        for t, l, o_, u_ in zip(tiempo, low, obj, up):
            l_draw = y_inf_low if not np.isfinite(l) else l
            u_draw = y_inf_up if not np.isfinite(u_) else u_
            alto = u_draw - l_draw

            ax.add_patch(plt.Rectangle((t - bar_width / 2, l_draw), bar_width, alto,
                                       facecolor='skyblue', alpha=0.4, edgecolor=None, zorder=1))

            if np.isfinite(l):
                ax.plot([t - bar_width / 2, t + bar_width / 2], [l, l], color='red', linewidth=2.5, zorder=2)
            else:
                ax.plot([t - bar_width / 4, t + bar_width / 4], [y_inf_low, y_inf_low],
                        color='red', linewidth=2.5, zorder=2)
                ax.text(t, y_inf_low - 0.015 * range_span, '∞', color='red', fontsize=12,
                        ha='center', va='top')

            if np.isfinite(u_):
                ax.plot([t - bar_width / 2, t + bar_width / 2], [u_, u_], color='red', linewidth=2.5, zorder=2)
            else:
                ax.plot([t - bar_width / 4, t + bar_width / 4], [y_inf_up, y_inf_up],
                        color='red', linewidth=2.5, zorder=2)
                ax.text(t, y_inf_up + 0.015 * range_span, '∞', color='red', fontsize=12,
                        ha='center', va='bottom')

            ax.plot([t - marker_width / 2, t + marker_width / 2], [o_, o_],
                    color='black', linewidth=2.5, zorder=3)

        ax.set_title(f"{nombre_var} - Tipo: {tipo}", fontsize=14)
        ax.set_ylabel("Valor coeficiente", fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.5)

    axs[-1].set_xlabel("Año", fontsize=12)
    plt.tight_layout()
    plt.show()
    return rangos

def graficar_un_escenario(informacion, variable, escenario, anios):
    """
    Genera un gráfico de barras apiladas que muestra la cantidad de autobuses por tipo en un escenario específico a lo largo de varios años.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable: str, nombre de la variable que contiene los datos de los autobuses.
    :param escenario: str, nombre del escenario a graficar.
    :param anios: list, lista de años para los cuales se desea realizar el análisis.
    """
    # Preparación de los datos
    df = informacion[escenario][variable]
    tipos_autobuses = {tipo: [] for tipo in df['Tipo de autobus'].unique()}

    for anio in anios:
        df_anio = df[df['Periodo'] == anio]
        count_by_tipo = df_anio.groupby('Tipo de autobus')['z'].sum()

        for tipo in tipos_autobuses.keys():
            tipos_autobuses[tipo].append(count_by_tipo.get(tipo, 0))

    # Creación del gráfico
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.5
    bar_positions = np.arange(len(anios))

    # Paleta de colores pastel
    colores_pastel = [
        '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
        '#D5BAFF', '#FFBAED', '#BAFFD5', '#B3FFBA', '#BADFFF'
    ]

    bottom = np.zeros(len(anios))
    for i, (tipo, counts) in enumerate(tipos_autobuses.items()):
        bars = ax.bar(bar_positions, counts, bar_width, bottom=bottom, label=tipo.replace('_', '-')[2:], color=colores_pastel[i % len(colores_pastel)])
        bottom += np.array(counts)
        # Añadir etiquetas con la cantidad en el centro de cada sección
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(count)}', ha='center', va='center', fontsize=11)

    # Configuración de los ejes y título
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(anios)
    ax.set_xlabel('Años',fontsize=15)
    ax.set_ylabel('Cantidad de autobuses disponibles',fontsize=15)

    # Leyenda fuera del gráfico, en la parte inferior y centrada en una fila
    handles, labels = ax.get_legend_handles_labels()
    labels = [i.replace('CT','TL') for i in labels]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.15), frameon=False,
              ncol=len(tipos_autobuses), fontsize=12.5)
    plt.text(0.11, -0.192, 'Tipos de autobús:', transform=plt.gca().transAxes, fontsize=14, ha='center', va='center')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    nom_grafico = 'Buses disponibles Caso Base.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')

    plt.show()

def graficar_un_escenario_cargadores_contenedores(informacion, variable_cargadores, variable_contenedores, escenario, anios):
    """
    Genera un gráfico de barras apiladas que muestra la cantidad de cargadores y contenedores por tipo en un escenario específico a lo largo de varios años.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable_cargadores: str, nombre de la variable que contiene los datos de los cargadores.
    :param variable_contenedores: str, nombre de la variable que contiene los datos de los contenedores.
    :param escenario: str, nombre del escenario a graficar.
    :param anios: list, lista de años para los cuales se desea realizar el análisis.
    """
    # Preparación de los datos
    df_cargadores = informacion[escenario][variable_cargadores]
    df_contenedores = informacion[escenario][variable_contenedores]
    tipos_cargadores = {tipo: [] for tipo in df_cargadores['Tipo de cargador'].unique()}
    tipos_cargadores['Contenedores'] = []

    for anio in anios:
        df_anio_cargadores = df_cargadores[df_cargadores['Periodo'] == anio]
        count_by_tipo_cargadores = df_anio_cargadores.groupby('Tipo de cargador')['v'].sum()

        for tipo in tipos_cargadores.keys():
            if tipo != 'Contenedores':
                tipos_cargadores[tipo].append(count_by_tipo_cargadores.get(tipo, 0))

        df_anio_contenedores = df_contenedores[df_contenedores['Periodo'] == anio]
        count_contenedores = df_anio_contenedores['v_tilde'].sum()
        tipos_cargadores['Contenedores'].append(count_contenedores)

    # Creación del gráfico
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.5
    bar_positions = np.arange(len(anios))

    # Paleta de colores pastel
    colores_pastel = [
        '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
        '#D5BAFF', '#FFBAED', '#BAFFD5', '#B3FFBA', '#BADFFF', '#E0BBE4'
    ]

    bottom = np.zeros(len(anios))
    for i, (tipo, counts) in enumerate(tipos_cargadores.items()):
        if i == len(tipos_cargadores.keys())-1:
            bars = ax.bar(bar_positions, counts, bar_width, bottom=bottom, label=tipo.replace('_', ' '), color=colores_pastel[i % len(colores_pastel)])
        else:
            bars = ax.bar(bar_positions, counts, bar_width, bottom=bottom, label=tipo.replace('_', ' ')[1:], color=colores_pastel[i % len(colores_pastel)])
        
        bottom += np.array(counts)
        # Añadir etiquetas con la cantidad en el centro de cada sección
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(count)}', ha='center', va='center', fontsize=11)

    # Configuración de los ejes y título
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(anios)
    ax.set_xlabel('Años',fontsize=15)
    ax.set_ylabel('Cantidad de cargadores y contenedores disponibles',fontsize=15)

    # Leyenda fuera del gráfico, en la parte inferior y centrada en una fila
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.15), ncol=len(tipos_cargadores), fontsize=12,frameon=False)
    #plt.text(0.07, -0.183, 'Tipo de cargador: ', transform=plt.gca().transAxes, fontsize=14, ha='center', va='center')
    
    #ax.legend(handles, ['CC', 'CD', 'CTL', 'Contenedor'], loc='upper left',bbox_to_anchor=(0.02, 0.97),
    #          title= 'Tipo de cargador', fontsize=12,frameon=False,title_fontsize=14)
    
    ax.legend(handles, ['CC', 'CD', 'CTL', 'Contenedor'], loc='upper center',bbox_to_anchor=(0.55, -0.14),
              ncol=4, fontsize=12,frameon=False,title_fontsize=14)
    
    plt.text(0.31, -0.172, 'Tipos de cargador:', transform=plt.gca().transAxes, fontsize=14, ha='center', va='center')
    
    #ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.15), frameon=False,
    #          ncol=len(tipos_autobuses), fontsize=12.5)
    
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    nom_grafico = 'Carg y Cont disponibles Caso Base.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()

def graficar_autobuses_por_tipo_varios_anios_subplots(informacion, variable, anios):
    """
    Genera gráficos de barras que muestran la cantidad de autobuses por tipo en diferentes escenarios para varios años,
    organizados en una disposición de matriz de subplots.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable: str, nombre de la variable que contiene los datos de los autobuses.
    :param anios: list, lista de años para los cuales se desea realizar el análisis.
    """
    # Preparación de los datos
    num_anios = len(anios)
    num_cols = 3 if num_anios == 5 else 5
    num_rows = (num_anios + num_cols - 1) // num_cols  # Calcula el número de filas necesarias

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharex=False if num_anios == 5 else True)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Paleta de colores pastel
    colores_pastel = [
        '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
        '#D5BAFF', '#FFBAED', '#BAFFD5', '#B3FFBA', '#BADFFF'
    ]

    bar_width = 0.5  # Definir el ancho de las barras

    # Recorrer cada año y generar un subplot correspondiente
    for idx, anio in enumerate(anios):
        ax = axes[idx // num_cols, idx % num_cols]

        escenarios = []
        tipos_autobuses = {}
        for escenario, data in informacion.items():
            if variable in data:
                df = data[variable]
                # Filtrar por año y sumar los valores de 'z' para cada tipo de autobús
                df_anio = df[df['Periodo'] == anio]
                count_by_tipo = df_anio.groupby('Tipo de autobus')['z'].sum()

                escenarios.append(escenario)
                for tipo, count in count_by_tipo.items():
                    if tipo not in tipos_autobuses:
                        tipos_autobuses[tipo] = [0] * len(informacion)  # Inicializar con ceros
                    tipos_autobuses[tipo][escenarios.index(escenario)] = count

        # Creación del gráfico para el año actual
        bar_positions = range(len(escenarios))
        bottom = [0] * len(escenarios)
        for i, (tipo, counts) in enumerate(tipos_autobuses.items()):
            bars = ax.bar(bar_positions, counts, bar_width, bottom=bottom, label=tipo.replace('_', '-')[2:], color=colores_pastel[i % len(colores_pastel)])
            bottom = [j + k for j, k in zip(bottom, counts)]
            # Añadir etiquetas con la cantidad en el centro de cada sección
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(count)}', ha='center', va='center', fontsize=9)

        # Cambiar el formato de las etiquetas de los escenarios
        etiquetas_escenarios = []
        for i in escenarios:
            if int(i.split(' ')[1]) != 0:
                esc = i.split(' ')[1]
                etiquetas_escenarios.append(f'$\\mathcal{{E}}_{{{esc}}}$')
            else:
                etiquetas_escenarios.append('Caso\nbase')

        if num_anios == 5:
            if idx in [2, 3, 4]:  # Muestra las etiquetas en el tercer, cuarto y quinto subplot
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(etiquetas_escenarios, fontsize=14)
            else:
                ax.set_xticks([])
        else:
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(etiquetas_escenarios, fontsize=12)
            
        ax.set_title(f'{anio}')

    # Etiquetas comunes y leyenda
    fig.text(0.5, 0.04, 'Escenarios', ha='center', va='center',fontsize=14)
    fig.text(0, 0.5, 'Cantidad de autobuses disponibles', ha='center', va='center', rotation='vertical',fontsize=14)

    # Leyenda fuera de los subplots
    handles, labels = ax.get_legend_handles_labels()
    labels = [i.replace('CT','TL') for i in labels]
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=8, title='Tipo de Autobús')
    if num_anios == 5:
        fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.75, 0.2),
               ncol=1, fontsize=13,frameon=False,title='Tipos de autobús',title_fontsize=14)
    else:
        fig.legend(handles, labels, loc='lower center',ncol=len(tipos_autobuses), fontsize=14,frameon=False, bbox_to_anchor=(0.57, -0.05))
        #ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.15), frameon=False,
        #      ncol=len(tipos_autobuses), fontsize=12.5)
        #plt.text(-5.4, -0.3, 'Tipos de autobús:', fontsize=14, ha='center', va='center')
              
        fig.text(-4.15, -0.335, 'Tipos de autobús: ', transform=plt.gca().transAxes, fontsize=14, ha='center', va='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # Ocultar subplots vacíos
    if num_anios == 5:
        
        for i in range(num_anios, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])
    
        nom_grafico = 'Buses disponibles escenarios 5per.pdf'
    else:
        nom_grafico = 'Buses disponibles escenarios 10per.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()

def graficar_inversion_acumulada_por_escenario(informacion, variable):
    """
    Genera un gráfico de líneas que muestra la inversión acumulada por escenario a lo largo de los años en millones de dólares,
    agrupando las líneas que tienen exactamente los mismos valores en cada año (redondeando a dos decimales).

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable: str, nombre de la variable que contiene los datos de la inversión.
    """
    escenarios = list(informacion.keys())
    periodos = set()
    datos_escenarios = {}

    # Recopilar datos y periodos
    for escenario, data in informacion.items():
        if variable in data:
            df = data[variable]
            periodos.update(df['Periodo'])
            datos_escenarios[escenario] = df

    # Ordenar los periodos
    periodos = sorted(periodos)
    
    # Inicializar diccionarios para las inversiones acumuladas por escenario
    inversiones_acumuladas = {}

    for escenario in escenarios:
        if escenario in datos_escenarios:
            df = datos_escenarios[escenario]
            periodos_escenario = sorted(df['Periodo'].unique())
            inversiones_acumuladas[escenario] = []
            acumulada = 0
            for periodo in periodos_escenario:
                inversion = df[df['Periodo'] == periodo]['b'].values
                inversion = inversion[0] if len(inversion) > 0 else 0
                acumulada += inversion / 1_000_000
                # Redondear a dos decimales antes de almacenar la inversión acumulada
                inversiones_acumuladas[escenario].append((periodo, round(acumulada, 2)))

    # Agrupar escenarios que comparten los mismos valores de inversión acumulada
    series_graficadas = {}  # Diccionario para agrupar los escenarios por sus series

    for escenario, inversiones in inversiones_acumuladas.items():
        # Verificamos que las series sean exactamente idénticas en cada periodo (redondeadas)
        valores_inversiones = tuple(inversion for _, inversion in inversiones)

        # Agrupar escenarios que tienen exactamente los mismos valores de inversión acumulada
        if valores_inversiones in series_graficadas:
            series_graficadas[valores_inversiones].append(escenario)
        else:
            series_graficadas[valores_inversiones] = [escenario]

    # Crear el gráfico de líneas
    fig, ax = plt.subplots(figsize=(12, 8))
    colores_lineas = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
        '#9edae5', '#f7b6d2'
    ]

    if len(colores_lineas) < len(series_graficadas):
        raise ValueError("Hay más series de escenarios que colores únicos disponibles. Se necesitan más colores únicos.")

    # Primero calculamos las series que resultan de la agrupación
    series_grafico = []
    for valores_inversiones, escenarios_equivalentes in series_graficadas.items():
        periodos_escenario = [inversion[0] for inversion in inversiones_acumuladas[escenarios_equivalentes[0]]]
        nombre_escenarios = ', '.join([f'$\\mathcal{{E}}_{{{esc.split(" ")[1]}}}$' if esc.split(" ")[1] != '0' else 'Caso base' for esc in escenarios_equivalentes])
        series_grafico.append((periodos_escenario, valores_inversiones, nombre_escenarios))

    # Luego graficamos las series
    for idx, (periodos_escenario, valores_inversiones, nombre_escenarios) in enumerate(series_grafico):
        ax.plot(periodos_escenario, valores_inversiones, label=nombre_escenarios, 
                color=colores_lineas[idx % len(colores_lineas)], linestyle='-', marker='o', markersize=4)

    # Añadir líneas horizontales en millones de dólares con estilo de resaltador
    ax.axhline(y=12, color='red', linestyle='-', linewidth=3, alpha=0.3)
    ax.axhline(y=16, color='red', linestyle='-', linewidth=3, alpha=0.3)

    # Configuración de los ticks del eje X para mostrar todos los años disponibles en los datos
    ax.set_xticks(periodos)
    ax.set_xticklabels([int(p) for p in periodos], fontsize=12)
    ax.set_xlabel('Años', fontsize=14)
    ax.set_ylabel('Costos de compras acumulados (en millones de USD)', fontsize=14)

    # Ubicar la leyenda dentro del gráfico, en la esquina inferior derecha con dos columnas
    ax.legend(title='Escenarios', loc='center right', ncol=1, fontsize=14, title_fontsize=14, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    nom_grafico = 'Inversion Acumulada por escenario periodos {}.pdf'.format(len(escenarios))
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()

def graficar_cargadores_y_contenedores_por_tipo_varios_anios_subplots(informacion, variable_cargadores, variable_contenedores, anios):
    """
    Genera gráficos de barras que muestran la cantidad de cargadores y contenedores por tipo en diferentes escenarios para varios años,
    organizados en una disposición de matriz de subplots.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable_cargadores: str, nombre de la variable que contiene los datos de los cargadores.
    :param variable_contenedores: str, nombre de la variable que contiene los datos de los contenedores.
    :param anios: list, lista de años para los cuales se desea realizar el análisis.
    :param reducir_espacio: bool, si es True, reduce el espacio entre las barras a la mitad y solo grafica el caso base y los escenarios 10 y 11.
    """
    num_anios = len(anios)
    #num_cols = 3 if not reducir_espacio or num_anios != 11 else 4
    num_cols = 3 if num_anios == 5 else 5
    num_rows = (num_anios + num_cols - 1) // num_cols  # Calcula el número de filas necesarias

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharex=False if num_anios == 5 else True)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Paleta de colores pastel
    colores_pastel = [
        '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
        '#D5BAFF', '#FFBAED', '#BAFFD5', '#B3FFBA', '#BADFFF', '#E0BBE4'
    ]

    bar_width = 0.4  # Definir el ancho de las barras

    # Identificar todos los tipos posibles de cargadores y contenedores
    tipos_cargadores_totales = set()
    for escenario, data in informacion.items():
        tipos_cargadores_totales.update(data[variable_cargadores]['Tipo de cargador'].unique())
    tipos_cargadores_totales = sorted(tipos_cargadores_totales)  # Asegurar un orden consistente
    tipos_cargadores_totales.append('Contenedores')  # Añadir 'Contenedores' al final

    # Recorrer cada año y generar un subplot correspondiente
    for idx, anio in enumerate(anios):
        ax = axes[idx // num_cols, idx % num_cols]

        escenarios = []
        tipos_cargadores = {tipo: [0] * len(informacion) for tipo in tipos_cargadores_totales}

        for escenario, data in informacion.items():
            escenarios.append(escenario)  # Asegurar que el escenario se agrega antes de usarlo
            
            df = data[variable_cargadores]
            # Filtrar por año y sumar los valores de 'v' para cada tipo de cargador
            df_anio = df[df['Periodo'] == anio]
            count_by_tipo = df_anio.groupby('Tipo de cargador')['v'].sum()

            for tipo, count in count_by_tipo.items():
                tipos_cargadores[tipo][escenarios.index(escenario)] = count
            
            df_contenedores = data[variable_contenedores]
            # Filtrar por año y sumar los valores de 'v_tilde' para los contenedores
            df_anio_contenedores = df_contenedores[df_contenedores['Periodo'] == anio]
            count_contenedores = df_anio_contenedores['v_tilde'].sum()

            tipos_cargadores['Contenedores'][escenarios.index(escenario)] = count_contenedores

        # Creación del gráfico para el año actual
        bar_positions = np.arange(len(escenarios))
        bottom = np.zeros(len(escenarios))
        for i, (tipo, counts) in enumerate(tipos_cargadores.items()):
            bars = ax.bar(bar_positions, counts, bar_width, bottom=bottom,
                          label=tipo.replace('1_','').replace('2_','').replace('3_','').replace('_', ' '),
                          color=colores_pastel[i % len(colores_pastel)])
            bottom += np.array(counts)
            # Añadir etiquetas con la cantidad en el centro de cada sección
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(count)}', ha='center', va='center', fontsize=8)

        # Cambiar el formato de las etiquetas de los escenarios
        etiquetas_escenarios = []
        for i in escenarios:
            if int(i.split(' ')[1]) != 0:
                esc = i.split(' ')[1]
                etiquetas_escenarios.append(f'$\\mathcal{{E}}_{{{esc}}}$')
            else:
                etiquetas_escenarios.append('Caso\nbase')
        
        if num_anios == 5:
            if idx in [2, 3, 4]:  # Muestra las etiquetas en el tercer, cuarto y quinto subplot
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(etiquetas_escenarios, fontsize=14)
            else:
                ax.set_xticks([])
        else:
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(etiquetas_escenarios, fontsize=13)

        ax.set_title(f'{anio}')

    # Ocultar subplots vacíos
    if num_anios == 5:
        for i in range(num_anios, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

    # Etiquetas comunes y leyenda
    fig.text(0.5, 0.04, 'Escenarios', ha='center', va='center', fontsize = 14)
    fig.text(-0.01, 0.5, 'Cantidad de cargadores y contenedores disponibles', ha='center', va='center', rotation='vertical', fontsize = 14)

    # Leyenda fuera de los subplots
    handles, labels = ax.get_legend_handles_labels()
    labels=['CC','CD','CTL','Contenedor']
    if num_anios == 5:
        fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.75, 0.25),
               ncol=1, fontsize=13,frameon=False,title='Tipos de cargadores',title_fontsize=14)
    else:
        fig.legend(handles, labels, loc='lower center',ncol=len(tipos_cargadores), fontsize=13,frameon=False, bbox_to_anchor=(0.5, -0.04))
        #ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.15), frameon=False,
        #      ncol=len(tipos_autobuses), fontsize=12.5)
        #plt.text(-5.4, -0.3, 'Tipos de autobús:', fontsize=14, ha='center', va='center')
              
        fig.text(-3.4, -0.335, 'Tipos de cargadores:', transform=plt.gca().transAxes, fontsize=14, ha='center', va='center')
    

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if num_anios == 5:
        nom_grafico = 'Carg y Cont disp por esc y anio 5per.pdf'
    else:
        nom_grafico = 'Carg y Cont disp por esc y anio 10per.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()

def graficar_costos_por_tipo(informacion, variable):
    """
    Genera un gráfico de barras apiladas que muestra los costos por tipo en diferentes escenarios.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable: str, nombre de la variable que contiene los datos de los costos.
    """
    escenarios = list(informacion.keys())
    tipos_costos = {}

    # Preparación de los datos
    for escenario, data in informacion.items():
        if variable in data:
            df = data[variable]
            df = df[df.index.str.contains('Costo') & ~df.index.str.contains('Costo total')]
            
            tipos_costos[escenario] = df.squeeze().to_dict()

    # Creación del gráfico de barras apiladas
    fig, ax = plt.subplots(figsize=(12, 8))
    #colores = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D5BAFF', '#FFBAED']
    colores = ['#FFDFBA','#FFB3BA' , '#FFFFBA', '#BAFFC9', '#BAE1FF','#d3d3d3']
    colores.append('#E0BBE4')  # Añadir un color extra para los costos de fin de horizonte

    bar_width = 0.5
    bar_positions = range(len(escenarios))
    bottom = [0] * len(escenarios)

    tipos_gastos = ['Costos de buses comprados', 'Costos de cargadores comprados', 'Costos de contenedores comprados', 'Costos operativos', 'Costos por demanda','Costos de media vida']
    nombres_tipos_gastos = ['Compras autobuses', 'Compras cargadores', 'Compras contenedores', 'Operativos', 'Potencia eléctrica','Media vida']

    # Lista para almacenar el costo total por barra
    total_costos = [0] * len(escenarios)

    # Añadir los costos normales primero
    for idx, tipo in enumerate(tipos_gastos):
        valores = [tipos_costos[escenario].get(tipo, 0) for escenario in escenarios]
        bars = ax.bar(bar_positions, valores, bar_width, bottom=bottom, label=nombres_tipos_gastos[idx], color=colores[idx % len(colores)])
        bottom = [i + j for i, j in zip(bottom, valores)]
        
        # Sumar los costos para cada barra
        total_costos = [total + valor for total, valor in zip(total_costos, valores)]

        # Añadir etiquetas con los valores en el centro de cada sección
        for bar, valor in zip(bars, valores):
            if valor > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{valor:.2f}', ha='center', va='center', fontsize=8)

    # Añadir los costos por fin de horizonte en la parte más alta de la barra
    valores_fin_horizonte = [tipos_costos[escenario].get('Costos por fin de horizonte', 0) for escenario in escenarios]
    bars = ax.bar(bar_positions, valores_fin_horizonte, bar_width, bottom=bottom, label='Fin de horizonte', color=colores[-1])

    # Sumar los costos de fin de horizonte para cada barra
    total_costos = [total + valor for total, valor in zip(total_costos, valores_fin_horizonte)]

    # Añadir etiquetas con los valores de costos de fin de horizonte
    for bar, valor in zip(bars, valores_fin_horizonte):
        if valor > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{valor:.2f}', ha='center', va='center', fontsize=9)

    # Añadir los costos totales encima de cada barra
    for idx, total in enumerate(total_costos):
        ax.text(bar_positions[idx], bottom[idx] + valores_fin_horizonte[idx] + 0.2, f'{total:.2f}', ha='center', va='bottom', fontsize=10)

    # Cambiar el formato de las etiquetas de los escenarios
    etiquetas_escenarios = []
    for escenario in escenarios:
        if int(escenario.split(' ')[1]) != 0:
            esc = escenario.split(' ')[1]
            etiquetas_escenarios.append(f'$\\mathcal{{E}}_{{{esc}}}$')
        else:
            etiquetas_escenarios.append('Caso\nbase')

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(etiquetas_escenarios, fontsize=13)

    ax.set_xlabel('Escenarios',fontsize=13)
    ax.set_ylabel('Costos (en millones de USD)',fontsize=13)

    # Leyenda fuera de los subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, title='Costos',frameon=False)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    nom_grafico = 'Costos por esc.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()

def graficar_emisiones_por_contaminante(informacion, variable, contaminante):
    """
    Genera un gráfico de líneas que muestra las emisiones de un contaminante específico por período en diferentes escenarios,
    incluyendo las metas de emisión y las emisiones antes de la planificación. Se evita la superposición de líneas gráficamente
    y se añade la información de los escenarios que comparten la misma línea en la leyenda.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable: str, nombre de la variable que contiene los datos de las emisiones.
    :param contaminante: str, tipo de contaminante que se desea graficar.
    """
    escenarios = list(informacion.keys())
    series_graficadas = {}  # Diccionario para agrupar los escenarios por sus series (líneas equivalentes)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Paleta de colores
    colores = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
        '#9edae5', '#f7b6d2'
    ]

    # Agrupar escenarios que comparten los mismos valores de emisión
    for idx, escenario in enumerate(escenarios):
        if variable in informacion[escenario]:
            df = informacion[escenario][variable]
            df_contaminante = df[df['Tipo Contaminante'] == contaminante]

            if not df_contaminante.empty:
                valores_emision = tuple(df_contaminante['Emisión Flota'].values)

                # Agrupar escenarios que tienen los mismos valores de emisión
                if valores_emision in series_graficadas:
                    series_graficadas[valores_emision].append(escenario)
                else:
                    series_graficadas[valores_emision] = [escenario]

    # Graficar cada serie única y crear una leyenda con todos los escenarios que comparten esa serie
    for idx, (valores_emision, escenarios_equivalentes) in enumerate(series_graficadas.items()):
        numero_escenarios = [esc.split(" ")[1] for esc in escenarios_equivalentes]
        nombre_escenario = ', '.join([f'$\\mathcal{{E}}_{{{num}}}$' if num != '0' else 'Caso base' for num in numero_escenarios])

        # Extraer el período de cualquier escenario de la serie (son iguales para todos)
        df_escenario = informacion[escenarios_equivalentes[0]][variable]
        df_contaminante = df_escenario[df_escenario['Tipo Contaminante'] == contaminante]

        # Graficar la línea para la serie única
        ax.plot(df_contaminante['Periodo'], valores_emision, label=nombre_escenario, 
                color=colores[idx % len(colores)], linestyle='-', marker='o', markersize=5)

    # Obtener datos para las metas de emisión y la emisión antes de la planificación
    df_cb = informacion['Escenario 0'][variable]
    df_cb_contaminante = df_cb[df_cb['Tipo Contaminante'] == contaminante]

    # Graficar las metas de emisión y la emisión antes de la planificación
    if not df_cb_contaminante.empty:
        ax.plot(df_cb_contaminante['Periodo'], df_cb_contaminante['Meta de emisión'], label='Máx emisiones', color='#FF7171', linestyle='--')
        ax.plot(df_cb_contaminante['Periodo'], df_cb_contaminante['Emisión Flota antes de la planificación'], label='Emisión 2024', color='gray', linestyle=':')

    # Ajustar el nombre del contaminante
    if contaminante == 'NOX':
        nom_cont = r'$NO_{x}$'
    elif contaminante == 'PM25':
        nom_cont = r'$PM_{2.5}$'
    elif contaminante == 'SO2':
        nom_cont = r'$SO_{2}$'
    else:
        nom_cont = contaminante

    # Configuración de los ejes y título
    ax.set_xlabel('Años', fontsize=14)
    ax.set_ylabel(f'Emisiones de {nom_cont} (gramos)', fontsize=14)

    # Asegurar que los años en el eje X se muestren como enteros
    periodos = df_cb_contaminante['Periodo'].unique()  # Extraer los años (periodos)
    plt.xticks(ticks=periodos, labels=[int(p) for p in periodos], fontsize=12)

    # Configuración de la leyenda
    handles, labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles, labels, loc='center right', ncol=1, fontsize=12, bbox_to_anchor=(1, 0.6),
                     title='Escenarios', title_fontsize=14, frameon=False)
    ax.add_artist(leg1)

    plt.tight_layout()

    # Guardar el gráfico
    nom_grafico = f'Emisiones de {contaminante}.pdf'
    plt.savefig(nom_grafico, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def graficar_costos_comparativos(informacion_original, informacion_CB, variable):
    """
    Genera un gráfico de barras apiladas que compara los costos por tipo en diferentes escenarios
    entre el conjunto de datos original y el conjunto donde las decisiones del caso base fueron fijadas.

    :param informacion_original: dict, estructura que contiene la información de los escenarios originales.
    :param informacion_CB: dict, estructura que contiene la información de los escenarios con el caso base fijado.
    :param variable: str, nombre de la variable que contiene los datos de los costos.
    """
    escenarios = list(informacion_original.keys())
    tipos_costos_original = {}
    tipos_costos_CB = {}

    # Preparación de los datos
    for escenario, data in informacion_original.items():
        if variable in data:
            df = data[variable]
            df = df[df.index.str.contains('Costo') & ~df.index.str.contains('Costo total')]
            tipos_costos_original[escenario] = df.squeeze().to_dict()

    for escenario, data in informacion_CB.items():
        if variable in data and escenario != 'Escenario 0':
            df = data[variable]
            df = df[df.index.str.contains('Costo') & ~df.index.str.contains('Costo total')]
            tipos_costos_CB[escenario] = df.squeeze().to_dict()

    # Creación del gráfico de barras apiladas comparativas
    fig, ax = plt.subplots(figsize=(14, 8))
    colores = ['#FFDFBA', '#FFB3BA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#d3d3d3']
    colores.append('#E0BBE4')  # Añadir un color extra para los costos de fin de horizonte

    bar_width = 0.35
    caso_base_width = bar_width * 1.5  # Hacer la barra del caso base un poco más ancha
    gap = 0.05  # Espacio entre las barras
    bar_positions = np.arange(1, len(escenarios) + 1)  # Posiciones para las barras

    tipos_gastos = ['Costos de buses comprados', 'Costos de cargadores comprados', 'Costos de contenedores comprados', 'Costos operativos', 'Costos por demanda', 'Costos de media vida']
    nombres_tipos_gastos = ['Autobuses comprados', 'Cargadores comprados', 'Contenedores comprados', 'Operativos', 'Demanda eléctrica', 'Media vida']

    # Graficar los costos para informacion_original
    bottom_original = np.zeros(len(escenarios))
    for idx, tipo in enumerate(tipos_gastos):
        valores_original = [tipos_costos_original[escenario].get(tipo, 0) for escenario in escenarios]
        for i, (escenario, valor) in enumerate(zip(escenarios, valores_original)):
            if escenario == 'Escenario 0':
                bars_original = ax.bar(bar_positions[i], valor, caso_base_width, bottom=bottom_original[i], label=nombres_tipos_gastos[idx], color=colores[idx % len(colores)])
            else:
                bars_original = ax.bar(bar_positions[i] - (bar_width + gap) / 2, valor, bar_width, bottom=bottom_original[i], color=colores[idx % len(colores)])
            bottom_original[i] += valor

            # Añadir etiquetas con los valores en el centro de cada sección
            if valor > 0:
                height = valor
                ax.text(bar_positions[i] - (bar_width + gap) / 2 if escenario != 'Escenario 0' else bar_positions[i],
                        bottom_original[i] - height / 2, f'{valor:.2f}', ha='center', va='center', fontsize=8)

    # Añadir los costos por fin de horizonte para informacion_original
    valores_fin_horizonte_original = [tipos_costos_original[escenario].get('Costos por fin de horizonte', 0) for escenario in escenarios]
    for i, (escenario, valor) in enumerate(zip(escenarios, valores_fin_horizonte_original)):
        if escenario == 'Escenario 0':
            bars_fin_horizonte_original = ax.bar(bar_positions[i], valor, caso_base_width, bottom=bottom_original[i], color=colores[-1])
        else:
            bars_fin_horizonte_original = ax.bar(bar_positions[i] - (bar_width + gap) / 2, valor, bar_width, bottom=bottom_original[i], color=colores[-1])

        # Añadir etiquetas con los valores de costos de fin de horizonte
        if valor > 0:
            height = valor
            ax.text(bar_positions[i] - (bar_width + gap) / 2 if escenario != 'Escenario 0' else bar_positions[i],
                    bottom_original[i] + height / 2, f'{valor:.2f}', ha='center', va='center', fontsize=9)

    # Graficar los costos para informacion_CB con hatches (tramas), excluyendo el caso base
    bottom_CB = np.zeros(len(escenarios))
    for idx, tipo in enumerate(tipos_gastos):
        valores_CB = [tipos_costos_CB.get(escenario, {}).get(tipo, 0) if escenario != 'Escenario 0' else 0 for escenario in escenarios]
        bars_CB = ax.bar(bar_positions + (bar_width + gap) / 2, valores_CB, bar_width, bottom=bottom_CB, color=colores[idx % len(colores)], hatch='/', edgecolor='#A9A9A9')
        bottom_CB += np.array(valores_CB)

        # Añadir etiquetas con los valores en el centro de cada sección
        for bar, valor in zip(bars_CB, valores_CB):
            if valor > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{valor:.2f}', ha='center', va='center', fontsize=8)

    # Añadir los costos por fin de horizonte para informacion_CB, excluyendo el caso base
    valores_fin_horizonte_CB = [tipos_costos_CB.get(escenario, {}).get('Costos por fin de horizonte', 0) if escenario != 'Escenario 0' else 0 for escenario in escenarios]
    bars_fin_horizonte_CB = ax.bar(bar_positions + (bar_width + gap) / 2, valores_fin_horizonte_CB, bar_width, bottom=bottom_CB, color=colores[-1], hatch='/', edgecolor='#A9A9A9')

    # Añadir etiquetas con los valores de costos de fin de horizonte para informacion_CB
    for bar, valor in zip(bars_fin_horizonte_CB, valores_fin_horizonte_CB):
        if valor > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{valor:.2f}', ha='center', va='center', fontsize=9)

    # Añadir barra para el escenario E_8 infactible
    for i, escenario in enumerate(escenarios):
        if escenario == 'Escenario 8' and escenario not in informacion_CB:
            ax.bar(bar_positions[i] + (bar_width + gap) / 2, 27.61, bar_width, bottom=bottom_CB[i], color='lightgray', hatch='/', edgecolor='#A9A9A9')
            ax.text(bar_positions[i] + (bar_width + gap) / 2, 27.61 / 2, 'Infactible', ha='center', va='center', fontsize=14,rotation=90)

    # Cambiar el formato de las etiquetas de los escenarios
    etiquetas_escenarios = []
    for escenario in escenarios:
        if int(escenario.split(' ')[1]) != 0:
            esc = escenario.split(' ')[1]
            etiquetas_escenarios.append(f'$\\mathcal{{E}}_{{{esc}}}$')
        else:
            etiquetas_escenarios.append('Caso\nbase')

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(etiquetas_escenarios, fontsize=13)

    ax.set_xlabel('Escenarios', fontsize=13)
    ax.set_ylabel('Costos (en millones de USD)', fontsize=13)

    # Mostrar costos totales sobre cada barra original y del caso base
    costos_totales_original = np.array([bottom_original[i] + valores_fin_horizonte_original[i] for i in range(len(escenarios))])
    costos_totales_CB = np.array([bottom_CB[i] + valores_fin_horizonte_CB[i] for i in range(len(escenarios))])

    # Añadir los textos de costos totales sobre las barras
    for i in range(len(escenarios)):
        if i == 0:
            ax.text(bar_positions[i], costos_totales_original[i] + 0.2, f'{costos_totales_original[i]:.2f}', ha='center', va='bottom', fontsize=9)
        elif i == 7:
            ax.text(bar_positions[i] - (bar_width + gap) / 2, costos_totales_original[i] + 0.2, f'{costos_totales_original[i]:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            # Costos totales sobre las barras originales
            ax.text(bar_positions[i] - (bar_width + gap) / 2, costos_totales_original[i] + 0.2, f'{costos_totales_original[i]:.2f}', ha='center', va='bottom', fontsize=9)
            # Costos totales sobre las barras del caso base
            ax.text(bar_positions[i] + (bar_width + gap) / 2, costos_totales_CB[i] + 0.2, f'{costos_totales_CB[i]:.2f}', ha='center', va='bottom', fontsize=9)

    # Leyenda fuera de los subplots, incluyendo fin de horizonte y explicación del hatch
    handles, labels = ax.get_legend_handles_labels()

    # Reorganizar los elementos de la leyenda
    fin_horizonte_patch = mpatches.Patch(color=colores[-1], label='Fin de horizonte')
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='#A9A9A9', hatch='///', label='Variación')

    # Insertar el patch del hatch al final
    handles = handles[:6] + [fin_horizonte_patch, hatch_patch]
    labels = labels[:6] + ['Fin de horizonte', 'Variación']
    
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, title='Costos', frameon=False)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    nom_grafico = 'Costos_comparativos_CB_vs_original.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()

def graficar_asignacion_autobuses_por_ruta(informacion, variable):
    """
    Genera un gráfico de barras apiladas que muestra la variación en la asignación de autobuses eléctricos a rutas
    en diferentes escenarios a lo largo de los años.

    :param informacion: dict, estructura que contiene la información de los escenarios.
    :param variable: str, nombre de la variable que contiene los datos de los autobuses eléctricos por ruta.
    """
    escenarios = list(informacion.keys())

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))
    axes = axes.flatten()

    # Colores para las rutas
    colores = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D5BAFF', '#FFBAED', '#E0BBE4']
    
    for idx, escenario in enumerate(escenarios):
        if variable in informacion[escenario]:
            df = informacion[escenario][variable]
            df = df[~df.index.str.contains('Total')]
            rutas = df.index
            anios = df.columns
            
            bottom = np.zeros(len(anios))
            ax = axes[idx]
            
            for i, ruta in enumerate(rutas):
                valores = df.loc[ruta].values
                # Asegurarse de que los valores de años coincidan con los escenarios
                bars = ax.bar(np.arange(len(valores)), valores, bottom=bottom[:len(valores)], label=ruta.replace('_',' '), color=colores[i % len(colores)])
                bottom[:len(valores)] += valores

                # Añadir etiquetas con los valores en el centro de cada sección
                for bar, valor in zip(bars, valores):
                    if valor > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{int(valor)}', ha='center', va='center', fontsize=9)
            
            # Configuración de los ejes
            numero_escenario = escenario.split(' ')[-1]
            nombre_escenario = 'Caso base' if numero_escenario == '0' else f'$\\mathcal{{E}}_{{{numero_escenario}}}$'
            ax.set_title(nombre_escenario, fontsize=16)
            ax.set_xticks(np.arange(len(anios)))
            ax.set_xticklabels(anios, fontsize=13)
    
    # Manejar los ejes vacíos si no hay suficientes escenarios
    for i in range(len(escenarios), len(axes)):
        fig.delaxes(axes[i])

    # Eje general
    fig.text(0.5, 0.07, 'Años', ha='center', va='center', fontsize=16)
    fig.text(0, 0.5, 'Número de Autobuses Eléctricos', ha='center', va='center', rotation='vertical', fontsize=20)

    # Leyenda común
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.7, 0.155),
               ncol=2, fontsize=13,frameon=False,title='Rutas',title_fontsize=14)
    #fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=11, title='Rutas', frameon=False)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    nom_grafico = 'Asignacion_autobuses_por_ruta_escenarios.pdf'
    plt.savefig(nom_grafico, format=nom_grafico.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()
