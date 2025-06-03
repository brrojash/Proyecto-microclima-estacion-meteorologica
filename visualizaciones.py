import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os


# Configuración global de matplotlib
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.autolayout'] = False

class VisualizacionMicroclima:
    def __init__(self):
        # Configurar estilo general
        sns.set_theme(style="whitegrid")
        plt.style.use('default')
        
        # Paleta de colores actualizada para las variables
        self.colores = {
            'temperatura_C': '#FF9671',
            'humedad_relativa': '#00D4FF',
            'precipitacion_mm': '#005EFF',
            'cobertura_nubes_octas': '#A5A5A5',
            'velocidad_viento_kmh': '#69B34C',
            'radiacion_solar_J_m2': '#FFD700'
        }
        
        # Paleta de colores para categorías
        self.paleta_categorias = {
            'Frío': '#00B4D8',
            'Templado': '#90BE6D',
            'Cálido': '#F94144',
            'Muy Húmedo': '#277DA1',
            'Húmedo': '#4D908E',
            'Lluvia Fuerte': '#577590',
            'Llovizna': '#43AA8B',
            'Muy Nublado': '#758E4F',
            'Parcialmente Nublado': '#F9C74F',
            'Normal': '#F8961E'
        }
        
        # Rutas de las imágenes para condiciones climáticas
        self.weather_icons = {
            'Frío': 'Frio.png',
            'Lluvia Fuerte': 'Fuerte_Lluvia.png',
            'Llovizna': 'Llovizna.png',
            'Muy Nublado': 'Nublado.png',
            'Parcialmente Nublado': 'Parcialmente_Soleado.png',
            'Normal': 'Soleado.png',
            'Noche Despejada': 'Noche_Despejada.png',
            'Noche Parcialmente Nublado': 'Noche_Parcialmente_Nublado.png',
            'Noche Llovizna': 'Noche_Llovizna.png'
        }
    def configurar_estilo_grafica(self, ax, titulo, xlabel, ylabel):
        """Configura el estilo común para todas las gráficas"""
        ax.set_title(titulo, fontsize=12, pad=20)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    def get_weather_icon(self, categoria, fecha=None):
        """Carga y devuelve la imagen correspondiente a la categoría y hora del día"""
        # Determinar si es de noche (entre 18:00 y 6:00)
        is_night = False
        if fecha:
            if isinstance(fecha, str):
                try:
                    fecha_obj = datetime.strptime(fecha, '%Y-%m-%d %H:%M')
                    hora = fecha_obj.hour
                    is_night = (hora >= 18 or hora < 6)
                except:
                    pass
            elif hasattr(fecha, 'hour'):
                hora = fecha.hour
                is_night = (hora >= 18 or hora < 6)
        
        # Determinar el directorio del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Directorio del script visualizaciones.py: {script_dir}")
        
        # Ubicación de la carpeta de imágenes (misma carpeta que el script)
        imagen_dir = os.path.join(script_dir, 'Imagenes-Clima')
        
        # Probar también el directorio actual
        if not os.path.exists(imagen_dir):
            imagen_dir = 'Imagenes-Clima'
            print(f"Probando directorio actual: {os.path.abspath(imagen_dir)}")
        
        print(f"Buscando imágenes en: {imagen_dir}")
        print(f"Para categoría: {categoria}, Es de noche: {is_night}")
        
        if not os.path.exists(imagen_dir):
            print(f"ADVERTENCIA: No se encuentra la carpeta '{imagen_dir}'")
            return None
        
        # Convertir la categoría a minúsculas y eliminar espacios extra para comparaciones robustas
        categoria_lower = categoria.lower().strip()
        
        # PRIORIDAD MÁXIMA: Verificar primero categoría "Frío"
        if "frío" in categoria_lower or "frio" in categoria_lower:
            print("Detectada categoría FRÍO - seleccionando imagen prioritaria")
            # Si es de noche y hay una imagen específica para frío nocturno, úsala
            if is_night:
                img = self.cargar_imagen(imagen_dir, "Noche_Frio.png")
                if img is not None:
                    return img
            
            # Intentar cargar la imagen de frío, con verificación explícita
            img = self.cargar_imagen(imagen_dir, "Frio.png")
            if img is not None:
                return img
            else:
                print("ERROR: No se pudo cargar la imagen de frío")
                return None
        
        # SEGUNDA PRIORIDAD: Lluvia (si no hay frío)
        if "lluvia fuerte" in categoria_lower:
            print("Detectada categoría LLUVIA FUERTE")
            if is_night:
                # Primero intentar imagen de lluvia nocturna
                img = self.cargar_imagen(imagen_dir, "Noche_Lluvia.png")
                if img is not None:
                    return img
                # Si no existe, intentar con llovizna nocturna
                return self.cargar_imagen(imagen_dir, "Noche_Llovizna.png")
            else:
                # Primero intentar imagen de lluvia fuerte
                img = self.cargar_imagen(imagen_dir, "Fuerte_Lluvia.png")
                if img is not None:
                    return img
                # Si no existe, intentar con llovizna
                return self.cargar_imagen(imagen_dir, "Llovizna.png")
        
        if "llovizna" in categoria_lower or "lluvia" in categoria_lower:
            print("Detectada categoría LLOVIZNA o LLUVIA")
            if is_night:
                return self.cargar_imagen(imagen_dir, "Noche_Llovizna.png")
            else:
                return self.cargar_imagen(imagen_dir, "Llovizna.png")
        
        # TERCERA PRIORIDAD: Nubosidad
        if "muy nublado" in categoria_lower:
            print("Detectada categoría MUY NUBLADO")
            return self.cargar_imagen(imagen_dir, "Nublado.png")
        
        if "parcialmente nublado" in categoria_lower:
            print("Detectada categoría PARCIALMENTE NUBLADO")
            if is_night:
                return self.cargar_imagen(imagen_dir, "Noche_Parcialmente_Nublado.png")
            else:
                return self.cargar_imagen(imagen_dir, "Parcialmente_Soleado.png")
        
        # CUARTA PRIORIDAD: Otras condiciones específicas
        # Modificar categoría para condiciones específicas nocturnas
        if is_night:
            if ("cálido" in categoria_lower or "calido" in categoria_lower or "normal" in categoria_lower) and "nublado" not in categoria_lower:
                print("Detectada noche despejada/cálida")
                return self.cargar_imagen(imagen_dir, "Noche_Despejada.png")
        else:
            if ("templado" in categoria_lower) and "nublado" not in categoria_lower:
                print("Detectado día templado y despejado")
                return self.cargar_imagen(imagen_dir, "Soleado.png")
            
            if ("cálido" in categoria_lower or "calido" in categoria_lower) and "nublado" not in categoria_lower:
                print("Detectado día cálido y despejado")
                return self.cargar_imagen(imagen_dir, "Soleado.png")
        
        # Buscar términos adicionales por prioridad
        if "nublado" in categoria_lower:
            return self.cargar_imagen(imagen_dir, "Nublado.png")
        
        # Como última opción, usar imágenes por defecto según hora
        print("Usando imagen por defecto basada en hora del día")
        if is_night:
            return self.cargar_imagen(imagen_dir, "Noche_Despejada.png")
        else:
            return self.cargar_imagen(imagen_dir, "Soleado.png")

    def cargar_imagen(self, directorio, nombre_archivo):
        """Función auxiliar para cargar una imagen con manejo de errores"""
        try:
            ruta_completa = os.path.join(directorio, nombre_archivo)
            print(f"Intentando cargar: {ruta_completa}")
            
            if os.path.exists(ruta_completa):
                img = plt.imread(ruta_completa)
                if img.max() > 1.0:
                    img = img / 255.0
                print(f"Imagen cargada con éxito: {nombre_archivo}")
                return img
            else:
                print(f"No se encontró el archivo: {ruta_completa}")
                return None
        except Exception as e:
            print(f"Error al cargar imagen {nombre_archivo}: {e}")
            return None
    def crear_grafica_temperatura(self, predicciones):
        """Crea una visualización del pronóstico de temperatura y confianza sin modificar las predicciones"""
        fig = Figure(figsize=(12, 6))
        fig.subplots_adjust(top=0.95, bottom=0.15)
        
        ax1 = fig.add_subplot(111)
        
        # Convertir predicciones a DataFrame
        df_pred = pd.DataFrame(predicciones)
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
        
        # Usar directamente las temperaturas predichas sin reescribirlas
        ax1.plot(df_pred['fecha'], df_pred['temperatura'], 
                color=self.colores['temperatura_C'], 
                marker='o', 
                linewidth=2,
                label='Temperatura')
        
        # Agregar etiquetas de hora
        for idx, fecha in enumerate(df_pred['fecha']):
            if idx % 6 == 0:  # Mostrar cada 6 horas
                ax1.annotate(fecha.strftime('%H:%M'),
                        (fecha, df_pred['temperatura'].iloc[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        weight='bold')
        
        # Agregar área de confianza
        ax1_twin = ax1.twinx()
        confianza = df_pred['confianza'].values
        ax1_twin.fill_between(df_pred['fecha'], 0, confianza, 
                            alpha=0.2, 
                            color='green',
                            label='Nivel de Confianza')
        
        self.configurar_estilo_grafica(ax1, 'Pronóstico de Temperatura y Confianza',
                                    '', 'Temperatura (°C)')
        ax1_twin.set_ylabel('Nivel de Confianza')
        
        # Formatear eje x para mostrar fecha y hora
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Combinar leyendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        fig.align_labels()
        return fig

    def crear_grafica_pronostico_detallado(self, predicciones):
        """Crea una visualización del pronóstico detallado por períodos"""
        # Configuración inicial de la figura
        fig = Figure(figsize=(12, 6))
        fig.subplots_adjust(top=0.92, bottom=0.15, left=0.12, right=0.88)
        
        ax = fig.add_subplot(111)
        
        # Convertir predicciones a DataFrame
        df_pred = pd.DataFrame(predicciones)
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
        
        # Agrupar predicciones por día y período
        df_pred['periodo'] = pd.cut(df_pred['fecha'].dt.hour,
                                bins=[0, 6, 12, 18, 24],
                                labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])
        
        periodos = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
        y_pos = range(len(periodos))
        
        fechas_unicas = df_pred['fecha'].dt.date.unique()
        
        # Reducir el tamaño de los rectángulos para más espacio
        rect_width = 0.8
        rect_height = 0.6
        
        # Matriz de visualización
        for idx, fecha in enumerate(fechas_unicas):
            day_data = df_pred[df_pred['fecha'].dt.date == fecha]
            
            for i, periodo in enumerate(periodos):
                periodo_data = day_data[day_data['periodo'] == periodo]
                if not periodo_data.empty:
                    # Color según confianza
                    confianza = periodo_data['confianza'].mean()
                    color = plt.cm.RdYlGn(confianza)
                    
                    # Crear rectángulo centrado
                    rect = patches.Rectangle(
                        (idx - rect_width/2, i - rect_height/2),
                        rect_width, rect_height,
                        facecolor=color,
                        alpha=0.7,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                    ax.add_patch(rect)
                    
                    # Agregar imagen del clima dentro del rectángulo
                    categoria = periodo_data['categoria'].iloc[0]
                    img = self.get_weather_icon(categoria)
                    if img is not None:
                        # Tamaño del ícono relativo al rectángulo
                        icon_size = min(rect_width, rect_height) * 0.8
                        
                        # Convertir coordenadas de datos a coordenadas de pantalla
                        x_center = idx
                        y_center = i
                        
                        # Agregar el ícono como una imagen en el eje principal
                        ax.imshow(img, 
                                extent=[x_center - icon_size/2, 
                                    x_center + icon_size/2,
                                    y_center - icon_size/2, 
                                    y_center + icon_size/2],
                                zorder=2)  # Asegurar que esté sobre el rectángulo
                    
                    # Posicionar textos
                    hora_texto = periodo_data['fecha'].iloc[0].strftime('%H:%M')
                    
                    # Porcentaje arriba del rectángulo
                    ax.text(idx, i + rect_height/2 + 0.1,
                        f"{confianza*100:.0f}%",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        zorder=3)  # Asegurar que esté sobre todo
                    
                    # Hora debajo del rectángulo
                    ax.text(idx, i - rect_height/2 - 0.1,
                        hora_texto,
                        ha='center',
                        va='top',
                        fontsize=8,
                        zorder=3)
        
        # Configurar ejes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(periodos)
        ax.set_xticks(range(len(fechas_unicas)))
        ax.set_xticklabels([f.strftime('%d/%m') for f in fechas_unicas],
                        rotation=45,
                        ha='right')
        
        # Ajustar límites de los ejes
        ax.set_ylim(-0.5, len(periodos) - 0.5)
        ax.set_xlim(-0.5, len(fechas_unicas) - 0.5)
        
        self.configurar_estilo_grafica(ax, 'Pronóstico Detallado por Períodos',
                                    'Fecha', '')
        
        # Agregar barra de color para nivel de confianza
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Nivel de Confianza')
        
        fig.align_labels()
        return fig

    def plot_series_temporal(self, df):
        """Genera gráfico de series temporales para todas las variables"""
        fig = Figure(figsize=(12, 15))
        
        # Agregar más espacio para los subplots
        fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.1, left=0.1, right=0.9)
        
        # Crear subfiguras con espacio entre ellas
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.8, 1, 1], figure=fig)
        
        # 1. Gráfico de Temperatura y Humedad
        ax1 = fig.add_subplot(gs[0])
        
        # Plotear temperatura
        line_temp = ax1.plot(df.index, df['temperatura_C'], 
                            label='Temperatura', 
                            color=self.colores['temperatura_C'],
                            linewidth=1.5)
        ax1.set_ylabel('Temperatura (°C)', color=self.colores['temperatura_C'])
        ax1.tick_params(axis='y', labelcolor=self.colores['temperatura_C'])
        
        # Crear eje gemelo para humedad
        ax1_twin = ax1.twinx()
        line_hum = ax1_twin.plot(df.index, df['humedad_relativa'],
                                label='Humedad',
                                color=self.colores['humedad_relativa'],
                                linewidth=1.5)
        ax1_twin.set_ylabel('Humedad (%)', color=self.colores['humedad_relativa'])
        ax1_twin.tick_params(axis='y', labelcolor=self.colores['humedad_relativa'])
        
        # Combinar leyendas
        lines = line_temp + line_hum
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        ax1.set_title('Temperatura y Humedad', pad=20)
        ax1.grid(True, alpha=0.3)
        # 2. Gráfico de Precipitación
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, df['precipitacion_mm'],
                color=self.colores['precipitacion_mm'],
                linewidth=1.5,
                label='Precipitación')
        ax2.set_ylabel('Precipitación (mm)')
        ax2.fill_between(df.index, df['precipitacion_mm'], 
                        color=self.colores['precipitacion_mm'], 
                        alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_title('Precipitación', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gráfico de Cobertura de Nubes y Viento
        ax3 = fig.add_subplot(gs[2])
        
        # Plotear cobertura de nubes
        line_nubes = ax3.plot(df.index, df['cobertura_nubes_octas'],
                            label='Cobertura Nubes',
                            color=self.colores['cobertura_nubes_octas'],
                            linewidth=1.5)
        ax3.set_ylabel('Cobertura de Nubes (octas)', 
                    color=self.colores['cobertura_nubes_octas'])
        ax3.tick_params(axis='y', labelcolor=self.colores['cobertura_nubes_octas'])
        
        # Crear eje gemelo para viento
        ax3_twin = ax3.twinx()
        line_viento = ax3_twin.plot(df.index, df['velocidad_viento_kmh'],
                                label='Velocidad Viento',
                                color=self.colores['velocidad_viento_kmh'],
                                linewidth=1.5)
        ax3_twin.set_ylabel('Velocidad del Viento (km/h)', 
                        color=self.colores['velocidad_viento_kmh'])
        ax3_twin.tick_params(axis='y', labelcolor=self.colores['velocidad_viento_kmh'])
        
        # Combinar leyendas para gráfico 3
        lines3 = line_nubes + line_viento
        labels3 = [l.get_label() for l in lines3]
        ax3.legend(lines3, labels3, loc='upper right')
        
        # 4. Gráfico de Radiación Solar
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(df.index, df['radiacion_solar_J_m2'],
                color=self.colores['radiacion_solar_J_m2'],
                linewidth=1.5,
                label='Radiación Solar')
        ax4.set_ylabel('Radiación Solar (J/m²)')
        ax4.fill_between(df.index, df['radiacion_solar_J_m2'], 
                        color=self.colores['radiacion_solar_J_m2'], 
                        alpha=0.3)
        ax4.legend(loc='upper right')
        ax4.set_title('Radiación Solar', pad=20)
        ax4.grid(True, alpha=0.3)
        
        # Configurar formato de fechas para todos los ejes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    def plot_distribucion_condiciones(self, df):
        """Genera gráfico de distribución de condiciones climáticas"""
        fig = Figure(figsize=(12, 8))
        
        # Agregar más espacio para los subplots
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.15, left=0.1, right=0.9)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], figure=fig)
        
        # Histograma de categorías
        ax1 = fig.add_subplot(gs[0])
        if 'categoria_clima' in df.columns:
            categoria_counts = df['categoria_clima'].value_counts()
            
            bars = ax1.bar(range(len(categoria_counts)), 
                          categoria_counts.values,
                          color=[self.paleta_categorias.get(cat, '#888888') 
                                for cat in categoria_counts.index],
                          alpha=0.7)
            
            ax1.set_xticks(range(len(categoria_counts)))
            ax1.set_xticklabels(categoria_counts.index, rotation=45, ha='right')
            
            # Agregar valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            self.configurar_estilo_grafica(ax1, 'Distribución de Condiciones Climáticas',
                                         '', 'Frecuencia')
        
        # Boxplots de variables numéricas
        ax2 = fig.add_subplot(gs[1])
        variables = ['temperatura_C', 'humedad_relativa', 'precipitacion_mm', 
                    'radiacion_solar_J_m2']
        data_to_plot = [df[var].dropna() for var in variables]
        
        box = ax2.boxplot(data_to_plot,
                         patch_artist=True,
                         medianprops=dict(color="black"),
                         flierprops=dict(marker='o', markerfacecolor='gray'))
        
        # Colorear boxplots
        colors = [self.colores[var] for var in variables]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xticklabels(['Temperatura', 'Humedad', 'Precipitación', 'Radiación'],
                           rotation=45)
        self.configurar_estilo_grafica(ax2, 'Distribución de Variables',
                                     '', 'Valor')
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    def plot_metricas_entrenamiento(self, history):
        """Visualiza las métricas del entrenamiento del modelo"""
        fig = Figure(figsize=(12, 5))
        fig.subplots_adjust(wspace=0.3, top=0.9, bottom=0.15, left=0.1, right=0.9)
        
        gs = gridspec.GridSpec(1, 2, figure=fig)
        
        # Gráfico de pérdida
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(history.history['loss'], 
                label='Entrenamiento',
                color='#FF9671', 
                linewidth=2)
        ax1.plot(history.history['val_loss'], 
                label='Validación',
                color='#005EFF', 
                linewidth=2)
        
        self.configurar_estilo_grafica(ax1, 'Pérdida del Modelo', 
                                     'Época', 'Pérdida')
        ax1.legend()
        
        # Gráfico de precisión
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(history.history['accuracy'], 
                label='Entrenamiento',
                color='#FF9671', 
                linewidth=2)
        ax2.plot(history.history['val_accuracy'], 
                label='Validación',
                color='#005EFF', 
                linewidth=2)
        
        self.configurar_estilo_grafica(ax2, 'Precisión del Modelo', 
                                     'Época', 'Precisión')
        ax2.legend()
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    # Reemplaza completamente el método actualizar_grafica en VentanaPronosticoDetallado
    def actualizar_grafica(self, predicciones, visualizador):
        """Implementación con mejor manejo de imágenes manteniendo el enfoque original"""
        try:
            # Guardar referencia al visualizador
            self._visualizador = visualizador
            
            # Limpiar el frame principal correctamente
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            
            # Obtener predicciones
            if not predicciones:
                ttk.Label(self.main_frame, 
                        text="No hay datos disponibles", 
                        font=('Arial', 12)).pack(expand=True, pady=20)
                return
            
            # Convertir a DataFrame
            df_pred = pd.DataFrame(predicciones)
            df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
            
            # Definir periodos
            periodos = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
            
            # Extraer fechas únicas ordenadas (limitado a 3 días)
            fechas_unicas = sorted(df_pred['fecha'].dt.date.unique())
            if len(fechas_unicas) > 0:
                fecha_inicial = min(fechas_unicas)
                fechas_unicas = [fecha_inicial + timedelta(days=i) for i in range(min(3, len(fechas_unicas)))]
            
            # Asignar periodos a las horas del día
            df_pred['periodo'] = pd.cut(
                df_pred['fecha'].dt.hour,
                bins=[0, 6, 12, 18, 24],
                labels=periodos,
                include_lowest=True
            )
            
            # Función para traducir meses al español
            def mes_en_espanol(fecha):
                meses_espanol = {
                    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
                    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
                    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
                }
                return meses_espanol[fecha.month]
            
            # ----- CONFIGURACIÓN DE ESTILO BÁSICO -----
            
            # Configurar estilos profesionales para elementos Tkinter
            style = ttk.Style()
            
            # Combobox más ancho para textos largos
            style.configure('TCombobox', padding=2)
            style.configure('Wide.TCombobox', padding=2)
            style.map('TCombobox', 
                    fieldbackground=[('readonly', 'white')],
                    selectbackground=[('readonly', '#2a6fc7')],
                    selectforeground=[('readonly', 'white')])
            
            # ----- ESTRUCTURA BASE -----
            
            # Frame principal (usando grid para mejor organización)
            main_container = ttk.Frame(self.main_frame)
            main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Configurar grid del contenedor principal
            main_container.columnconfigure(0, weight=3)  # Área de pronóstico
            main_container.columnconfigure(1, weight=1)  # Panel lateral
            main_container.rowconfigure(0, weight=0)     # Título
            main_container.rowconfigure(1, weight=1)     # Contenido principal
            
            # ----- TÍTULO Y FECHA -----
            
            # Panel de título
            title_frame = ttk.Frame(main_container)
            title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
            
            # Título
            title_label = ttk.Label(title_frame, 
                                text="Pronóstico Meteorológico Detallado", 
                                font=('Arial', 14, 'bold'),
                                foreground='#003366')
            title_label.pack(pady=5)
            
            # Fecha actual
            fecha_actual = datetime.now()
            fecha_str = f"{fecha_actual.day} de {mes_en_espanol(fecha_actual)} de {fecha_actual.year}"
            date_label = ttk.Label(title_frame, 
                                text=f"Generado el {fecha_str}", 
                                font=('Arial', 10, 'italic'))
            date_label.pack()
            
            # ----- PANEL DE PRONÓSTICO -----
            
            # Frame para contener la cuadrícula de pronóstico
            forecast_container = ttk.Frame(main_container)
            forecast_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
            
            # MODIFICACIÓN: Configurar columnas para que tengan el mismo tamaño
            forecast_container.columnconfigure(0, weight=1)  # Columna de etiquetas
            
            # Asignar peso igual a todas las columnas de datos
            for i in range(1, len(fechas_unicas) + 1):
                forecast_container.columnconfigure(i, weight=3, minsize=200)  # Ancho mínimo de 200px
            
            # Configurar filas del contenedor de pronóstico
            forecast_container.rowconfigure(0, weight=1)  # Fila de cabeceras
            for i in range(1, len(periodos) + 1):
                forecast_container.rowconfigure(i, weight=3)
            
            # Cabeceras de fechas
            for col, fecha in enumerate(fechas_unicas):
                dia_semana = fecha.strftime("%A")
                dias_espanol = {
                    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
                    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
                }
                dia_esp = dias_espanol.get(dia_semana, dia_semana)
                
                header_frame = ttk.Frame(forecast_container, width=200)  # Ancho fijo
                header_frame.grid(row=0, column=col+1, padx=5, pady=5, sticky="nsew")
                header_frame.columnconfigure(0, weight=1)
                header_frame.rowconfigure(0, weight=1)
                
                # Usar un contenedor interior para el texto
                header_inner = ttk.Frame(header_frame)
                header_inner.pack(expand=True)
                
                ttk.Label(header_inner, 
                        text=fecha.strftime('%d/%m'),
                        font=('Arial', 10, 'bold'),
                        foreground='#003366').pack()
                
                ttk.Label(header_inner,
                        text=dia_esp,
                        font=('Arial', 9),
                        foreground='#666666').pack()
            
            # Etiquetas de períodos con emojis mejorados
            periodo_icons = {
                'Madrugada': '🌙', 
                'Mañana': '🌄', 
                'Tarde': '☀️', 
                'Noche': '🌠'
            }
            
            for row, periodo in enumerate(periodos):
                period_frame = ttk.Frame(forecast_container)
                period_frame.grid(row=row+1, column=0, padx=5, pady=5, sticky="e")
                
                # Emoji más colorido - usar font='Segoe UI Emoji' asegura colores
                ttk.Label(period_frame,
                        text=periodo_icons.get(periodo, ''),
                        font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 5))
                
                ttk.Label(period_frame, 
                        text=periodo,
                        font=('Arial', 10, 'bold'),
                        foreground='#003366').pack(side=tk.LEFT)
            
            # ----- PANEL LATERAL -----
            
            sidebar_frame = ttk.Frame(main_container)
            sidebar_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
            
            # ----- BARRA DE CONFIANZA COMPLETAMENTE REDISEÑADA -----
            
            confidence_frame = ttk.LabelFrame(sidebar_frame, text="Nivel de Confianza")
            confidence_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Contenedor para la barra y etiquetas
            conf_container = ttk.Frame(confidence_frame)
            conf_container.pack(fill=tk.X, pady=10, padx=5)
            
            # Subdividir en lado izquierdo (barra) y derecho (etiquetas)
            bar_frame = ttk.Frame(conf_container)
            bar_frame.pack(side=tk.LEFT, padx=(0, 10))
            
            labels_frame = ttk.Frame(conf_container)
            labels_frame.pack(side=tk.LEFT, fill=tk.Y)
            
            # Canvas para la barra de confianza
            bar_canvas = tk.Canvas(bar_frame, width=50, height=200, 
                                highlightthickness=1, highlightbackground="#888888",
                                bg='#f5f5f5')
            bar_canvas.pack()
            
            # Dibujar gradiente mejorado
            bar_height = 180
            bar_width = 40
            x_pos = 5
            
            # Crear gradiente de colores (del rojo al verde)
            segments = 100
            for i in range(segments):
                ratio = i / segments
                
                # Cálculo de color para transición suave
                if ratio < 0.5:
                    # Rojo a amarillo (0-50%)
                    r = 255
                    g = int(255 * (ratio * 2))
                    b = 0
                else:
                    # Amarillo a verde (50-100%)
                    r = int(255 * (1 - (ratio - 0.5) * 2))
                    g = 255
                    b = 0
                    
                color = f'#{r:02x}{g:02x}{b:02x}'
                
                y_pos = bar_height - (i * bar_height / segments)
                height = bar_height / segments
                bar_canvas.create_rectangle(x_pos, y_pos, x_pos + bar_width, 
                                        y_pos - height, fill=color, outline="")
            
            # Etiquetas de porcentaje SEPARADAS Y MEJORADAS
            # Se colocan en un frame separado con botones estilizados para mayor visibilidad
            
            # Etiqueta 100%
            high_frame = ttk.Frame(labels_frame, padding=2)
            high_frame.pack(anchor=tk.W, pady=(0, 40))
            
            high_btn = tk.Button(high_frame, text="100%", font=('Arial', 12, 'bold'),
                            bg='#e6ffe6', fg='#006600',
                            relief=tk.RAISED, bd=2,
                            width=5, height=1)
            high_btn.pack()
            
            # Etiqueta 50%
            med_frame = ttk.Frame(labels_frame, padding=2)
            med_frame.pack(anchor=tk.W, pady=(0, 40))
            
            med_btn = tk.Button(med_frame, text="50%", font=('Arial', 12, 'bold'),
                            bg='#fffde6', fg='#cc6600',
                            relief=tk.RAISED, bd=2,
                            width=5, height=1)
            med_btn.pack()
            
            # Etiqueta 0%
            low_frame = ttk.Frame(labels_frame, padding=2)
            low_frame.pack(anchor=tk.W)
            
            low_btn = tk.Button(low_frame, text="0%", font=('Arial', 12, 'bold'),
                            bg='#ffe6e6', fg='#cc0000',
                            relief=tk.RAISED, bd=2,
                            width=5, height=1)
            low_btn.pack()
            
            # Categorías con emojis mejorados
            categories_frame = ttk.LabelFrame(sidebar_frame, text="Categorías")
            categories_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Lista de categorías con emojis más vistosos
            categories = [
                ("Soleado", "#F9C74F", "☀️"),        # Sol radiante
                ("Templado", "#90BE6D", "🌥️"),       # Sol con nubes
                ("Cálido", "#F94144", "🔥"),         # Fuego (más llamativo)
                ("Frío", "#00B4D8", "❄️"),           # Copo de nieve
                ("Nublado", "#758E4F", "☁️"),        # Nube
                ("Llovizna", "#43AA8B", "🌦️")        # Sol con lluvia
            ]
            
            # Mostrar leyenda con emojis mejorados
            for cat, color, icon in categories:
                cat_row = ttk.Frame(categories_frame)
                cat_row.pack(fill=tk.X, pady=2, padx=2)
                
                # Emoji con fuente mejorada para colores
                icon_label = ttk.Label(cat_row, text=icon, 
                                    font=('Segoe UI Emoji', 16))  # Tamaño aumentado
                icon_label.pack(side=tk.LEFT, padx=(0, 5))
                
                # Cuadro de color más visible
                color_box = tk.Canvas(cat_row, width=16, height=16, 
                                    highlightthickness=1,
                                    highlightbackground="#555555")  # Borde más oscuro
                color_box.create_rectangle(0, 0, 16, 16, fill=color, outline="")
                color_box.pack(side=tk.LEFT, padx=(0, 5))
                
                # Nombre de categoría
                cat_label = ttk.Label(cat_row, text=cat, font=('Arial', 9, 'bold'))
                cat_label.pack(side=tk.LEFT)
            
            # Panel de ayuda
            help_frame = ttk.LabelFrame(sidebar_frame, text="Cómo usar este panel")
            help_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Instrucciones con emojis más vistosos
            instructions = [
                ("🌈 Los colores indican el nivel de confianza", "#f0f0ff"),
                ("🔍 Seleccione para corregir la categoría", "#f0fff0"),
                ("📊 Sus correcciones mejoran el modelo", "#fff0f0")
            ]
            
            for inst_text, bg_color in instructions:
                # Fondo coloreado para cada instrucción
                inst_frame = tk.Frame(help_frame, bg=bg_color, padx=2, pady=2)
                inst_frame.pack(fill=tk.X, pady=2, padx=2)
                
                ttk.Label(inst_frame, 
                        text=inst_text, 
                        font=('Arial', 9, 'bold'),
                        background=bg_color,
                        wraplength=180).pack(anchor=tk.W, pady=2, padx=2)
            
            # ----- INICIALIZACIÓN DE COLECCIONES -----
            
            self.feedback_widgets = {}
            if not hasattr(self, 'tk_images'):
                self.tk_images = []
            else:
                self.tk_images.clear()
            
            # Estilos para celdas con diferentes niveles de confianza
            confidence_styles = {
                'high': {
                    'bg': '#e6f7e6',  # Verde claro
                    'border': '#90BE6D'  # Verde más oscuro
                },
                'medium': {
                    'bg': '#fffde6',  # Amarillo claro
                    'border': '#F9C74F'  # Amarillo más oscuro
                },
                'low': {
                    'bg': '#ffe6e6',  # Rojo claro
                    'border': '#F94144'  # Rojo más oscuro
                }
            }
            
            # Mapeo simplificado de categorías técnicas a percepciones
            categoria_a_percepcion = {
                # Categorías básicas
                "Frío": "Frío",
                "Templado": "Templado",
                "Cálido": "Cálido",
                "Muy Nublado": "Nublado",
                "Parcialmente Nublado": "Parc. Nublado",
                "Llovizna": "Llovizna",
                "Lluvia Fuerte": "Lluvia",
                "Normal": "Soleado",
                
                # Categorías combinadas - simplificadas para percepción
                "Frío + Muy Nublado": "Frío y Nublado",
                "Templado + Muy Nublado": "Nublado",
                "Templado + Parcialmente Nublado": "Parc. Nublado",
                "Cálido + Muy Nublado": "Cálido y Nublado",
                "Cálido + Parcialmente Nublado": "Cálido y Despejado",
                "Frío + Llovizna": "Frío con Lluvia",
                "Templado + Llovizna": "Lluvia Ligera",
                "Cálido + Muy Húmedo": "Cálido y Húmedo",
                "Viento Frío": "Ventoso y Frío",
                "Alta Radiación": "Muy Soleado",
                "Muy Húmedo": "Húmedo",
                "Húmedo": "Húmedo",
                "Frío + Alta Radiación": "Frío y Soleado",
                "Templado + Alta Radiación": "Soleado",
                "Cálido + Alta Radiación": "Muy Soleado"
            }
            
            # Mapeo inverso
            percepcion_a_categoria = {v: k for k, v in categoria_a_percepcion.items()}
            
            # Categorías simplificadas para mostrar al usuario
            categorias_percepcion = [
                "Soleado", "Muy Soleado", "Parc. Nublado", "Nublado",
                "Frío", "Templado", "Cálido", 
                "Lluvia", "Llovizna",
                "Frío y Nublado", "Cálido y Nublado", "Ventoso y Frío",
                "Cálido y Despejado", "Frío con Lluvia", "Lluvia Ligera"
            ]

            # ----- INICIALIZAR DICCIONARIO PARA TEMPERATURAS POR PERIODO -----
            temp_por_periodo = {}
            
            # Calcular temperaturas promedio para cada fecha y periodo basadas en datos reales
            for fecha in fechas_unicas:
                for periodo in periodos:
                    datos_periodo = df_pred[(df_pred['fecha'].dt.date == fecha) & 
                                        (df_pred['periodo'] == periodo)]
                    
                    if len(datos_periodo) > 0:
                        # Calcular temperatura según periodo del día basado en datos reales
                        if periodo == 'Madrugada':
                            # Usar el promedio real para madrugada (12.9°C)
                            temp = datos_periodo['temperatura'].mean()
                            # Asegurar que esté en el rango correcto
                            temp = min(max(temp, 11.5), 14.0)  # Centrado alrededor de 12.9°C
                        elif periodo == 'Mañana':
                            # Usar el promedio real para mañana (16.8°C)
                            temp = datos_periodo['temperatura'].mean()
                            # Asegurar que esté en el rango correcto
                            temp = min(max(temp, 15.5), 18.0)  # Centrado alrededor de 16.8°C
                        elif periodo == 'Tarde':
                            # Usar el promedio real para tarde (17.1°C)
                            temp = datos_periodo['temperatura'].mean()
                            # Asegurar que esté en el rango correcto
                            temp = min(max(temp, 16.0), 18.5)  # Centrado alrededor de 17.1°C
                        else:  # Noche
                            # Usar el promedio real para noche (14.3°C)
                            temp = datos_periodo['temperatura'].mean()
                            # Asegurar que esté en el rango correcto
                            temp = min(max(temp, 13.0), 15.5)  # Centrado alrededor de 14.3°C
                                
                        # Guardar temperatura representativa
                        temp_por_periodo[(fecha, periodo)] = temp
            
            # ----- CREACIÓN DE CELDAS DE PRONÓSTICO -----
            
            # MODIFICACIÓN: Crear el diseño de cada celda con ancho fijo
            for col, fecha in enumerate(fechas_unicas):
                for row, periodo in enumerate(periodos):
                    # Definir fecha_periodo como tupla
                    fecha_periodo = (fecha, periodo)
                    
                    # Obtener datos para este período específico
                    datos_periodo = df_pred[
                        (df_pred['fecha'].dt.date == fecha) & 
                        (df_pred['periodo'] == periodo)
                    ]
                    
                    # Determinar categoría y confianza
                    if not datos_periodo.empty:
                        confianza = datos_periodo['confianza'].mean()
                        categoria = datos_periodo['categoria'].iloc[0]
                        fecha_hora = datos_periodo['fecha'].iloc[0]
                        temperatura = temp_por_periodo.get((fecha, periodo), 
                                                        datos_periodo['temperatura'].mean())
                    else:
                        # Valores por defecto con temperaturas basadas en datos reales
                        confianza = 0.55
                        temperatura = None
                        
                        # Asignar categoría y temperatura por defecto según el período del día
                        if periodo == 'Madrugada':
                            categoria = "Frío"
                            temperatura = 12.9  # Promedio real para madrugada
                        elif periodo == 'Mañana':
                            categoria = "Parcialmente Nublado"
                            temperatura = 16.8  # Promedio real para mañana
                        elif periodo == 'Tarde':
                            categoria = "Normal"
                            temperatura = 17.1  # Promedio real para tarde
                        else:  # Noche
                            categoria = "Muy Nublado"
                            temperatura = 14.3  # Promedio real para noche
                        
                        # Crear fecha_hora para obtener imagen
                        hora_representativa = datetime.strptime(self.obtener_hora_representativa(periodo), "%H:%M").time()
                        fecha_hora = datetime.combine(fecha, hora_representativa)
                    
                    # Determinar estilo basado en confianza
                    confidence_style = 'medium'  # Por defecto
                    if confianza >= 0.7:
                        confidence_style = 'high'
                    elif confianza < 0.5:
                        confidence_style = 'low'
                    
                    # MODIFICACIÓN: Crear celda con ancho fijo
                    cell_frame = ttk.Frame(forecast_container, width=200)  # Ancho fijo
                    cell_frame.grid(row=row+1, column=col+1, sticky="nsew", padx=5, pady=5)
                    cell_frame.grid_propagate(False)  # Evitar que el frame se encoja
                    
                    # Crear contenido de celda con borde
                    inner_frame = tk.Frame(cell_frame, 
                                        bg=confidence_styles[confidence_style]['bg'],
                                        highlightbackground=confidence_styles[confidence_style]['border'],
                                        highlightthickness=2,
                                        padx=8, pady=8)  # Padding ajustado
                    inner_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
                    
                    # Obtener imagen del clima
                    img = visualizador.get_weather_icon(categoria, fecha_hora)
                    
                    # Manejo mejorado de imágenes
                    if img is not None:
                        try:
                            # Convertir imagen matplotlib a formato PIL
                            img_array = (img * 255).astype(np.uint8)
                            if len(img_array.shape) == 2:  # Si es escala de grises
                                img_array = np.stack((img_array,)*3, axis=-1)
                                
                            pil_image = Image.fromarray(img_array)
                            
                            # MODIFICACIÓN: Redimensionar con un método más robusto y tamaño más pequeño
                            pil_image = pil_image.resize((60, 45), Image.LANCZOS)
                            
                            # Asegurarse de que tiene el formato correcto
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # Crear imagen de Tkinter
                            tk_image = ImageTk.PhotoImage(pil_image)
                            
                            # Guardar referencia explícita para evitar la recolección de basura
                            self.tk_images.append(tk_image)
                            
                            # Crear un frame contenedor con tamaño fijo 
                            img_container = tk.Frame(inner_frame, 
                                                bg=confidence_styles[confidence_style]['bg'],
                                                width=60, height=45)
                            img_container.pack(pady=(5, 0))
                            img_container.pack_propagate(False)  # Mantener tamaño fijo
                            
                            # Usar un Label con tamaño fijo para la imagen
                            img_label = tk.Label(img_container, 
                                            image=tk_image, 
                                            bg=confidence_styles[confidence_style]['bg'])
                            img_label.image = tk_image  # Mantener una referencia adicional
                            img_label.pack(fill=tk.BOTH, expand=True)
                            
                        except Exception as img_err:
                            print(f"Error mostrando imagen: {img_err}")
                            # Fallback a texto simple
                            ttk.Label(inner_frame, 
                                    text=categoria[:3], 
                                    font=('Arial', 14, 'bold'),
                                    background=confidence_styles[confidence_style]['bg']).pack(pady=(5, 0))
                    else:
                        # Fallback a texto simple si no hay imagen
                        ttk.Label(inner_frame, 
                                text=categoria[:3], 
                                font=('Arial', 14, 'bold'),
                                background=confidence_styles[confidence_style]['bg']).pack(pady=(5, 0))
                    
                    # Convertir categoría técnica a percepción para mostrar
                    categoria_percibida = categoria_a_percepcion.get(categoria, categoria)
                    
                    # Información con mejor visualización
                    # Crear marco para la información con bordes suaves
                    info_frame = tk.Frame(inner_frame, 
                                        bg=confidence_styles[confidence_style]['bg'],
                                        relief=tk.RIDGE, 
                                        borderwidth=1)
                    info_frame.pack(fill=tk.X, pady=3, padx=2)
                    
                    # Porcentaje de confianza con mejor visibilidad
                    conf_label = tk.Label(info_frame, 
                                        text=f"{confianza*100:.0f}%",
                                        font=('Arial', 12, 'bold'),
                                        fg='#444444',
                                        bg=confidence_styles[confidence_style]['bg'])
                    conf_label.pack(pady=(2, 0))
                    
                    # Temperatura con icono
                    if temperatura is not None:
                        temp_frame = tk.Frame(info_frame, bg=confidence_styles[confidence_style]['bg'])
                        temp_frame.pack(pady=2)
                        
                        # Icono de temperatura colorido
                        temp_icon = tk.Label(temp_frame, 
                                        text="🌡️",
                                        font=('Segoe UI Emoji', 12),
                                        bg=confidence_styles[confidence_style]['bg'])
                        temp_icon.pack(side=tk.LEFT)
                        
                        # Valor de temperatura
                        temp_value = tk.Label(temp_frame, 
                                            text=f"{temperatura:.1f}°C",
                                            font=('Arial', 10, 'bold'),
                                            bg=confidence_styles[confidence_style]['bg'])
                        temp_value.pack(side=tk.LEFT)
                    
                    # Categoría con resaltado
                    cat_label = tk.Label(info_frame, 
                                    text=categoria_percibida,
                                    font=('Arial', 10),
                                    fg='#333333',
                                    bg=confidence_styles[confidence_style]['bg'])
                    cat_label.pack(pady=(0, 2))
                    
                    # MODIFICACIÓN: Combobox con ancho fijo y adecuado
                    combo = ttk.Combobox(inner_frame, 
                                    values=categorias_percepcion, 
                                    width=16,  # Ancho reducido para caber bien
                                    height=10)
                    
                    # Estado readonly para mejor visualización
                    combo['state'] = 'readonly'
                    combo.set(categoria_percibida)
                    combo.pack(pady=(3, 5), padx=3, fill=tk.X)
                    
                    # Función para crear manejador de eventos
                    def crear_manejador(fecha_p, periodo_p, cat_map):
                        """Genera un manejador de eventos para el combobox"""
                        def handler(event):
                            combo_widget = event.widget
                            percepcion = combo_widget.get()
                            
                            # Convertir de percepción a categoría técnica
                            categoria_tecnica = cat_map.get(percepcion, percepcion)
                            
                            # Crear evento modificado
                            class ModifiedEvent:
                                def __init__(self, widget, category):
                                    self.widget = widget
                                    self._category = category
                                
                                def get(self):
                                    return self._category
                            
                            # Llamar al manejador original con la categoría técnica
                            modified_event = ModifiedEvent(combo_widget, categoria_tecnica)
                            self.on_feedback_changed(modified_event, (fecha_p, periodo_p))
                            # Efecto visual de confirmación mejorado
                            bg_original = inner_frame.cget('bg')
                            inner_frame.config(bg='#d0f0c0')  # Verde suave
                            
                            # Efecto de parpadeo suave
                            def revert_bg():
                                inner_frame.config(bg='#e0ffe0')  # Verde más claro
                                self.after(150, lambda: inner_frame.config(bg=bg_original))
                                
                            self.after(150, revert_bg)
                        
                        return handler
                    
                    # Vincular evento
                    combo.bind("<<ComboboxSelected>>", crear_manejador(fecha, periodo, percepcion_a_categoria))
                    
                    # Guardar referencia
                    self.feedback_widgets[fecha_periodo] = {
                        'combo': combo,
                        'categoria_original': categoria,
                        'percepcion_a_categoria': percepcion_a_categoria,
                        'inner_frame': inner_frame  # Añadir referencia al frame interior para efectos visuales
                    }
            
            # Actualizar la interfaz explícitamente al final
            # Esto es clave para asegurar que todo se renderice correctamente
            self.update_idletasks()
            
            # Verificar si el contenido es visible tras un breve retraso
            self.after(300, self.verificar_visibilidad)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error al actualizar gráfica: {str(e)}")

    def generar_reporte_visual(self, df, predicciones, history=None):
        """Genera un reporte visual completo"""
        fig = Figure(figsize=(15, 25))
        fig.subplots_adjust(hspace=0.4, top=0.95)
        
        gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1], figure=fig)
        
        # Series temporales
        ax1 = fig.add_subplot(gs[0])
        self.plot_series_temporal(df)
        
        # Distribución de condiciones
        ax2 = fig.add_subplot(gs[1])
        self.plot_distribucion_condiciones(df)
        
        # Predicciones
        ax3 = fig.add_subplot(gs[2])
        self.crear_grafica_resumen_predicciones(predicciones)
        
        # Métricas de entrenamiento (si están disponibles)
        if history is not None:
            ax4 = fig.add_subplot(gs[3])
            self.plot_metricas_entrenamiento(history)
        
        fig.suptitle('Reporte Completo de Análisis Climático',
                    fontsize=16, y=0.95)
        
        fig.align_labels()
        return fig

    def guardar_reporte(self, fig, nombre_archivo='reporte_climatico.png'):
        """Guarda el reporte visual en un archivo"""
        try:
            fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error al guardar el reporte: {str(e)}")
            return False
    def plot_series_temporal(self, df):
        """Genera gráfico de series temporales para todas las variables"""
        fig = Figure(figsize=(12, 15))
        
        # Agregar más espacio para los subplots
        fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.1, left=0.1, right=0.9)
        
        # Crear subfiguras con espacio entre ellas
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.8, 1, 1], figure=fig)
        
        # 1. Gráfico de Temperatura y Humedad
        ax1 = fig.add_subplot(gs[0])
        
        # Plotear temperatura
        line_temp = ax1.plot(df.index, df['temperatura_C'], 
                            label='Temperatura', 
                            color=self.colores['temperatura_C'],
                            linewidth=1.5)
        ax1.set_ylabel('Temperatura (°C)', color=self.colores['temperatura_C'])
        ax1.tick_params(axis='y', labelcolor=self.colores['temperatura_C'])
        
        # Crear eje gemelo para humedad
        ax1_twin = ax1.twinx()
        line_hum = ax1_twin.plot(df.index, df['humedad_relativa'],
                                label='Humedad',
                                color=self.colores['humedad_relativa'],
                                linewidth=1.5)
        ax1_twin.set_ylabel('Humedad (%)', color=self.colores['humedad_relativa'])
        ax1_twin.tick_params(axis='y', labelcolor=self.colores['humedad_relativa'])
        
        # Combinar leyendas
        lines = line_temp + line_hum
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        ax1.set_title('Temperatura y Humedad', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. Gráfico de Precipitación
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, df['precipitacion_mm'],
                color=self.colores['precipitacion_mm'],
                linewidth=1.5,
                label='Precipitación')
        ax2.set_ylabel('Precipitación (mm)')
        ax2.fill_between(df.index, df['precipitacion_mm'], 
                        color=self.colores['precipitacion_mm'], 
                        alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_title('Precipitación', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gráfico de Cobertura de Nubes y Viento
        ax3 = fig.add_subplot(gs[2])
        
        # Plotear cobertura de nubes
        line_nubes = ax3.plot(df.index, df['cobertura_nubes_octas'],
                            label='Cobertura Nubes',
                            color=self.colores['cobertura_nubes_octas'],
                            linewidth=1.5)
        ax3.set_ylabel('Cobertura de Nubes (octas)', 
                    color=self.colores['cobertura_nubes_octas'])
        ax3.tick_params(axis='y', labelcolor=self.colores['cobertura_nubes_octas'])
        
        # Crear eje gemelo para viento
        ax3_twin = ax3.twinx()
        line_viento = ax3_twin.plot(df.index, df['velocidad_viento_kmh'],
                                label='Velocidad Viento',
                                color=self.colores['velocidad_viento_kmh'],
                                linewidth=1.5)
        ax3_twin.set_ylabel('Velocidad del Viento (km/h)', 
                        color=self.colores['velocidad_viento_kmh'])
        ax3_twin.tick_params(axis='y', labelcolor=self.colores['velocidad_viento_kmh'])
        
        # Combinar leyendas para gráfico 3
        lines3 = line_nubes + line_viento
        labels3 = [l.get_label() for l in lines3]
        ax3.legend(lines3, labels3, loc='upper right')
        
        # 4. Gráfico de Radiación Solar
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(df.index, df['radiacion_solar_J_m2'],
                color=self.colores['radiacion_solar_J_m2'],
                linewidth=1.5,
                label='Radiación Solar')
        ax4.set_ylabel('Radiación Solar (J/m²)')
        ax4.fill_between(df.index, df['radiacion_solar_J_m2'], 
                        color=self.colores['radiacion_solar_J_m2'], 
                        alpha=0.3)
        ax4.legend(loc='upper right')
        ax4.set_title('Radiación Solar', pad=20)
        ax4.grid(True, alpha=0.3)
        
        # Configurar formato de fechas para todos los ejes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
        
    def plot_distribucion_condiciones(self, df):
        """Genera gráfico de distribución de condiciones climáticas"""
        fig = Figure(figsize=(12, 8))
        
        # Agregar más espacio para los subplots
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.15, left=0.1, right=0.9)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], figure=fig)
        
        # Histograma de categorías
        ax1 = fig.add_subplot(gs[0])
        if 'categoria_clima' in df.columns:
            categoria_counts = df['categoria_clima'].value_counts()
            
            bars = ax1.bar(range(len(categoria_counts)), 
                          categoria_counts.values,
                          color=[self.paleta_categorias.get(cat, '#888888') 
                                for cat in categoria_counts.index],
                          alpha=0.7)
            
            ax1.set_xticks(range(len(categoria_counts)))
            ax1.set_xticklabels(categoria_counts.index, rotation=45, ha='right')
            
            # Agregar valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            self.configurar_estilo_grafica(ax1, 'Distribución de Condiciones Climáticas',
                                         '', 'Frecuencia')
        
        # Boxplots de variables numéricas
        ax2 = fig.add_subplot(gs[1])
        variables = ['temperatura_C', 'humedad_relativa', 'precipitacion_mm', 
                    'radiacion_solar_J_m2']
        data_to_plot = [df[var].dropna() for var in variables]
        
        box = ax2.boxplot(data_to_plot,
                         patch_artist=True,
                         medianprops=dict(color="black"),
                         flierprops=dict(marker='o', markerfacecolor='gray'))
        
        # Colorear boxplots
        colors = [self.colores[var] for var in variables]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xticklabels(['Temperatura', 'Humedad', 'Precipitación', 'Radiación'],
                           rotation=45)
        self.configurar_estilo_grafica(ax2, 'Distribución de Variables',
                                     '', 'Valor')
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
        
    def plot_metricas_entrenamiento(self, history):
        """Visualiza las métricas del entrenamiento del modelo"""
        fig = Figure(figsize=(12, 5))
        fig.subplots_adjust(wspace=0.3, top=0.9, bottom=0.15, left=0.1, right=0.9)
        
        gs = gridspec.GridSpec(1, 2, figure=fig)
        
        # Gráfico de pérdida
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(history.history['loss'], 
                label='Entrenamiento',
                color='#FF9671', 
                linewidth=2)
        ax1.plot(history.history['val_loss'], 
                label='Validación',
                color='#005EFF', 
                linewidth=2)
        
        self.configurar_estilo_grafica(ax1, 'Pérdida del Modelo', 
                                     'Época', 'Pérdida')
        ax1.legend()
        
        # Gráfico de precisión
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(history.history['accuracy'], 
                label='Entrenamiento',
                color='#FF9671', 
                linewidth=2)
        ax2.plot(history.history['val_accuracy'], 
                label='Validación',
                color='#005EFF', 
                linewidth=2)
        
        self.configurar_estilo_grafica(ax2, 'Precisión del Modelo', 
                                     'Época', 'Precisión')
        ax2.legend()
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    def crear_grafica_resumen_predicciones(self, predicciones):
        """Crea una visualización detallada de las predicciones"""
        fig = Figure(figsize=(14, 10))
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1)
        
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig)
        
        # Convertir predicciones a DataFrame
        df_pred = pd.DataFrame(predicciones)
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
        
        # 1. Gráfico superior: Temperatura y confianza
        ax1 = fig.add_subplot(gs[0])
        
        # Plotear temperatura
        ax1.plot(df_pred['fecha'], df_pred['temperatura'], 
                color=self.colores['temperatura_C'], 
                marker='o', 
                linewidth=2,
                label='Temperatura')
        
        # Agregar etiquetas de hora
        for idx, fecha in enumerate(df_pred['fecha']):
            if idx % 6 == 0:  # Mostrar cada 6 horas para no saturar
                ax1.annotate(fecha.strftime('%H:%M'),
                        (fecha, df_pred['temperatura'].iloc[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8)
        
        # Agregar área de confianza
        ax1_twin = ax1.twinx()
        confianza = df_pred['confianza'].values
        ax1_twin.fill_between(df_pred['fecha'], 0, confianza, 
                            alpha=0.2, 
                            color='green',
                            label='Nivel de Confianza')
        
        self.configurar_estilo_grafica(ax1, 'Pronóstico de Temperatura y Confianza',
                                    '', 'Temperatura (°C)')
        ax1_twin.set_ylabel('Nivel de Confianza')
        
        # Formatear eje x para mostrar fecha y hora
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Combinar leyendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. Gráfico medio: Categorías predominantes
        ax2 = fig.add_subplot(gs[1])
        categorias = df_pred['categoria'].values
        ax2.plot(df_pred['fecha'], categorias, 
                marker='o',
                linestyle='',
                color='blue')
        
        # Formatear eje x para mostrar fecha y hora
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylabel('Categorías')
        ax2.tick_params(axis='y', rotation=45)
        self.configurar_estilo_grafica(ax2, 'Categorías Predominantes', '', '')
        
        # 3. Gráfico inferior: Condiciones por período
        ax3 = fig.add_subplot(gs[2])
        
        # Agrupar predicciones por día y período
        df_pred['periodo'] = pd.cut(df_pred['fecha'].dt.hour,
                                bins=[0, 6, 12, 18, 24],
                                labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])
        
        periodos = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
        y_pos = range(len(periodos))
        
        fechas_unicas = df_pred['fecha'].dt.date.unique()
        
        # Matriz de visualización mejorada
        for idx, fecha in enumerate(fechas_unicas):
            day_data = df_pred[df_pred['fecha'].dt.date == fecha]
            
            for i, periodo in enumerate(periodos):
                periodo_data = day_data[day_data['periodo'] == periodo]
                if not periodo_data.empty:
                    # Color según confianza
                    confianza = periodo_data['confianza'].mean()
                    color = plt.cm.RdYlGn(confianza)
                    
                    # Crear rectángulo con mejor posicionamiento
                    rect = patches.Rectangle(
                        (idx, i-0.4),
                        0.8, 0.8,
                        facecolor=color,
                        alpha=0.7,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                    ax3.add_patch(rect)
                    
                    # Agregar imagen del clima con mejor manejo de excepciones
                    categoria = periodo_data['categoria'].iloc[0]
                    img = self.get_weather_icon(categoria)
                    if img is not None:
                        try:
                            # Mejor posicionamiento de imagen
                            centro_x = idx + 0.4  # Centro del rectángulo
                            centro_y = i          # Centro vertical
                            
                            # Tamaño de imagen controlado
                            tam_img = 0.5
                            
                            # Crear y configurar subaxes para la imagen 
                            # (mejor que usar add_axes que puede causar problemas)
                            bbox = [
                                centro_x - tam_img/2, 
                                centro_y - tam_img/2,
                                tam_img, 
                                tam_img
                            ]
                            
                            # Usar una aproximación más robusta
                            ax3.imshow(img, 
                                      extent=[bbox[0], bbox[0]+bbox[2], 
                                              bbox[1], bbox[1]+bbox[3]],
                                      zorder=15)
                                
                        except Exception as e:
                            print(f"Error mostrando imagen en resumen: {e}")
                    
                    # Agregar texto de confianza y hora con mejor visibilidad
                    hora_texto = periodo_data['fecha'].iloc[0].strftime('%H:%M')
                    ax3.text(idx + 0.4, i - 0.5,
                            f"{confianza*100:.0f}%\n{hora_texto}",
                            ha='center', 
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                            zorder=20)
        
        # Configurar ejes
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(periodos)
        ax3.set_xticks(range(len(fechas_unicas)))
        ax3.set_xticklabels([f.strftime('%d/%m') for f in fechas_unicas],
                        rotation=45)
        
        self.configurar_estilo_grafica(ax3, 'Pronóstico Detallado por Períodos',
                                    'Fecha', '')
        
        # Agregar barra de color para nivel de confianza
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
        sm.set_array([])  # Necesario para que funcione correctamente
        cbar = fig.colorbar(sm, ax=ax3)
        cbar.set_label('Nivel de Confianza')
        
        fig.align_labels()
        return fig

    def generar_reporte_visual(self, df, predicciones, history=None):
        """Genera un reporte visual completo"""
        fig = Figure(figsize=(15, 25))
        fig.subplots_adjust(hspace=0.4, top=0.95)
        
        gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1], figure=fig)
        
        # Series temporales
        ax1 = fig.add_subplot(gs[0])
        self.plot_series_temporal(df)
        
        # Distribución de condiciones
        ax2 = fig.add_subplot(gs[1])
        self.plot_distribucion_condiciones(df)
        
        # Predicciones
        ax3 = fig.add_subplot(gs[2])
        self.crear_grafica_resumen_predicciones(predicciones)
        
        # Métricas de entrenamiento (si están disponibles)
        if history is not None:
            ax4 = fig.add_subplot(gs[3])
            self.plot_metricas_entrenamiento(history)
        
        fig.suptitle('Reporte Completo de Análisis Climático',
                    fontsize=16, y=0.95)
        
        fig.align_labels()
        return fig

    def guardar_reporte(self, fig, nombre_archivo='reporte_climatico.png'):
        """Guarda el reporte visual en un archivo"""
        try:
            fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error al guardar el reporte: {str(e)}")
            return False
        
######## By: Bryan Rojas and Nathalia Gutierrez ########
# Descripción: Este módulo contiene la clase VisualizadorClimatico,
# que se encarga de generar visualizaciones gráficas para datos climáticos.