import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import sys
import matplotlib.dates as mdates
import glob
import threading
from predictor_model import PrediccionMeteo
from visualizaciones import VisualizacionMicroclima
import seaborn as sns
import gc
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageTk
from estacion_meteorologica import EstacionMeteoApp  # Importar la clase principal
import tkinter as tk
#Integrar ventana de proceso de procesar Datos
from ProcesarDatosGUI import EstacionMeteorologicaGUI
# Integrar Dataset de 7 años

def integrar_datasets(ruta_historico, ruta_estacion_propia, ruta_salida):
    """Integra el dataset histórico con los datos de la estación meteorológica local"""
    # Cargar datasets
    df_historico = pd.read_csv(ruta_historico)
    df_estacion = pd.read_csv(ruta_estacion_propia)
    
    # Convertir fechas a datetime
    df_historico['fecha'] = pd.to_datetime(df_historico['fecha'])
    df_estacion['fecha'] = pd.to_datetime(df_estacion['fecha'])
    
    # Convertir radiacion_solar_wm2 a radiacion_solar_J_m2 si existe
    if 'radiacion_solar_wm2' in df_estacion.columns and 'radiacion_solar_J_m2' not in df_estacion.columns:
        print("Convirtiendo unidades de radiación solar de W/m² a J/m²...")
        # Para datos horarios, multiplicamos por 3600 segundos
        periodo_segundos = 3600
        df_estacion['radiacion_solar_J_m2'] = df_estacion['radiacion_solar_wm2'] * periodo_segundos
        # Eliminar la columna original
        df_estacion = df_estacion.drop('radiacion_solar_wm2', axis=1)
    
    # Asegurar que ambos datasets tengan las mismas columnas
    columnas_comunes = ['fecha', 'temperatura_C', 'humedad_relativa', 
                       'precipitacion_mm', 'cobertura_nubes_octas', 
                       'velocidad_viento_kmh', 'radiacion_solar_J_m2']
    
    # Verificar que todas las columnas existan
    for df, nombre in [(df_historico, "histórico"), (df_estacion, "estación")]:
        for col in columnas_comunes:
            if col != 'fecha' and col not in df.columns:
                print(f"Advertencia: La columna '{col}' no existe en el dataset {nombre}. Creando columna con valores NaN.")
                df[col] = float('nan')
    
    df_historico = df_historico[columnas_comunes]
    df_estacion = df_estacion[columnas_comunes]
    
    # Convertir a valores numéricos
    for col in columnas_comunes[1:]:
        df_historico[col] = pd.to_numeric(df_historico[col], errors='coerce')
        df_estacion = df_estacion.copy()  # Hacer una copia explícita
        df_estacion.loc[:, col] = pd.to_numeric(df_estacion[col], errors='coerce')
    
    # Establecer fechas como índice
    df_historico.set_index('fecha', inplace=True)
    df_estacion.set_index('fecha', inplace=True)
    
    # PUNTO CLAVE: Verificar el hueco temporal entre datasets
    print(f"Fecha final del histórico: {df_historico.index.max()}")
    print(f"Fecha inicial de estación: {df_estacion.index.min()}")
    diferencia_dias = (df_estacion.index.min() - df_historico.index.max()).days
    print(f"Diferencia entre datasets: {diferencia_dias} días")
    
    # Si hay un hueco temporal grande, generar un puente de datos para visualización
    if diferencia_dias > 30:
        print(f"ADVERTENCIA: Existe un hueco temporal de {diferencia_dias} días entre datasets")
        print("Creando puente temporal mejorado para mantener la continuidad...")
        
        fecha_inicio = df_historico.index.max() + pd.Timedelta(days=1)
        fecha_fin = df_estacion.index.min() - pd.Timedelta(days=1)
        
        # Crear fechas intermedias con mayor frecuencia (cada 6 horas en lugar de 15 días)
        fechas_puente = pd.date_range(
            start=fecha_inicio,
            end=fecha_fin,
            freq='6H'  # Frecuencia aumentada para mayor granularidad
        )
        
        if len(fechas_puente) > 0:
            # Crear dataframe puente con valores interpolados
            df_puente = pd.DataFrame(index=fechas_puente)
            
            # Definir factores estacionales por mes para ajustar las interpolaciones
            factores_estacionales = {
                # Mes: [factor_temperatura, factor_precipitacion, factor_humedad, factor_nubosidad]
                1: [-0.8, 0.2, 0.5, 0.3],   # Enero
                2: [-0.5, 0.3, 0.4, 0.3],   # Febrero
                3: [0.0, 0.8, 0.6, 0.5],    # Marzo
                4: [0.2, 1.2, 0.8, 0.7],    # Abril
                5: [0.3, 1.0, 0.7, 0.6],    # Mayo
                6: [0.2, 0.4, 0.5, 0.4],    # Junio
                7: [0.1, 0.3, 0.4, 0.3],    # Julio
                8: [0.0, 0.3, 0.5, 0.4],    # Agosto
                9: [0.1, 0.7, 0.6, 0.5],    # Septiembre
                10: [0.0, 1.3, 0.8, 0.7],   # Octubre
                11: [-0.2, 1.1, 0.9, 0.8],  # Noviembre
                12: [-0.5, 0.5, 0.6, 0.5]   # Diciembre
            }
            
            # Patrones horarios para variables como temperatura y radiación
            patrones_horarios = {
                'temperatura_C': {  # Ajuste horario para temperatura (en °C)
                    0: -1.5, 1: -2.0, 2: -2.5, 3: -3.0, 4: -3.0, 5: -2.5,  # Madrugada
                    6: -2.0, 7: -1.0, 8: 0.0, 9: 1.0, 10: 2.0, 11: 3.0,    # Mañana
                    12: 3.5, 13: 4.0, 14: 4.0, 15: 3.5, 16: 2.5, 17: 1.5,  # Tarde
                    18: 0.5, 19: 0.0, 20: -0.5, 21: -1.0, 22: -1.2, 23: -1.3  # Noche
                },
                'radiacion_solar_J_m2': {  # Factor multiplicativo para radiación
                    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.1,        # Madrugada
                    6: 0.2, 7: 0.4, 8: 0.6, 9: 0.8, 10: 0.9, 11: 1.0,      # Mañana
                    12: 1.0, 13: 0.95, 14: 0.9, 15: 0.8, 16: 0.6, 17: 0.4, # Tarde
                    18: 0.2, 19: 0.1, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0   # Noche
                }
            }
            
            # Calcular valores de referencia
            for col in columnas_comunes[1:]:
                # Obtener valores de referencia
                ultimos_datos_hist = df_historico[col].iloc[-48:]  # Últimas 48 horas del histórico
                primeros_datos_est = df_estacion[col].iloc[:48]   # Primeras 48 horas de la estación
                
                valor_inicial = ultimos_datos_hist.mean()  # Promedio en lugar del último valor
                valor_final = primeros_datos_est.mean()    # Promedio en lugar del primer valor
                
                # Variables para guardar valores interpolados
                valores_interpolados = []
                
                # Generar datos interpolados para cada fecha del puente
                for idx, fecha in enumerate(fechas_puente):
                    # Cálculo de progreso (de 0 a 1) a lo largo del período del puente
                    progreso = idx / (len(fechas_puente) - 1) if len(fechas_puente) > 1 else 0.5
                    
                    # Interpolación base con curva suavizada (función sigmoide en lugar de lineal)
                    # Esto hace que la transición sea más suave al inicio y al final
                    factor_sigmoid = 1 / (1 + np.exp(-10 * (progreso - 0.5)))
                    valor_base = valor_inicial + (valor_final - valor_inicial) * factor_sigmoid
                    
                    # Ajuste estacional según mes
                    mes = fecha.month
                    factores = factores_estacionales[mes]
                    
                    # Aplicar ajustes específicos por variable
                    if col == 'temperatura_C':
                        # Ajuste por hora del día
                        ajuste_hora = patrones_horarios['temperatura_C'].get(fecha.hour, 0)
                        # Ajuste estacional
                        ajuste_estacional = factores[0]
                        valor_ajustado = valor_base + ajuste_hora + ajuste_estacional
                        
                    elif col == 'precipitacion_mm':
                        # Ajuste estacional para precipitación
                        factor_precip = factores[1]
                        # Añadir variabilidad para precipitación (más realista)
                        variabilidad = np.random.exponential(0.5) if np.random.random() < 0.2 else 0
                        valor_ajustado = valor_base * factor_precip + variabilidad
                        valor_ajustado = max(0, valor_ajustado)  # Precipitación no puede ser negativa
                        
                    elif col == 'humedad_relativa':
                        # Ajuste estacional para humedad
                        factor_humedad = factores[2]
                        # Correlación inversa con temperatura
                        hora = fecha.hour
                        es_dia = 8 <= hora <= 17
                        ajuste_hora = -5 if es_dia else 5  # Menor humedad durante el día
                        valor_ajustado = valor_base * factor_humedad + ajuste_hora
                        valor_ajustado = min(max(valor_ajustado, 30), 100)  # Rango válido: 30-100%
                        
                    elif col == 'cobertura_nubes_octas':
                        # Ajuste estacional para nubosidad
                        factor_nubes = factores[3]
                        valor_ajustado = valor_base * factor_nubes
                        valor_ajustado = min(max(valor_ajustado, 0), 8)  # Rango válido: 0-8 octas
                        
                    elif col == 'radiacion_solar_J_m2':
                        # Radiación depende fuertemente de la hora
                        factor_hora = patrones_horarios['radiacion_solar_J_m2'].get(fecha.hour, 0)
                        # Valor típico de radiación para ese mes/hora
                        radiacion_tipica = 0 if factor_hora == 0 else 900000 * factor_hora
                        # Ajustar por nubosidad (correlación inversa)
                        nubosidad_estimada = 4  # Valor promedio si no tenemos dato real
                        factor_nubosidad = max(0, 1 - (nubosidad_estimada / 10))
                        valor_ajustado = radiacion_tipica * factor_nubosidad
                        
                    elif col == 'velocidad_viento_kmh':
                        # Velocidad del viento con patrón diurno
                        hora = fecha.hour
                        factor_hora = 1.2 if 10 <= hora <= 16 else 0.8  # Mayor velocidad en horas de sol
                        variabilidad = np.random.normal(0, 0.5)
                        valor_ajustado = valor_base * factor_hora + variabilidad
                        valor_ajustado = max(0, valor_ajustado)  # No puede ser negativa
                        
                    else:
                        # Para otras variables, usar interpolación simple
                        valor_ajustado = valor_base
                    
                    valores_interpolados.append(valor_ajustado)
                
                # Asignar los valores interpolados
                df_puente[col] = valores_interpolados
            
            print(f"Puente temporal mejorado creado con {len(df_puente)} puntos de datos")
            
            # Combinar los tres datasets
            df_combinado = pd.concat([df_historico, df_puente, df_estacion])
        else:
            # Si no hay suficiente espacio para crear puente, solo unir los datasets
            df_combinado = pd.concat([df_historico, df_estacion])
    else:
        # Si no hay un hueco grande, combinar normalmente
        df_combinado = pd.concat([df_historico, df_estacion])
    
    # Si hay solapamiento de fechas, priorizar datos de la estación propia
    df_combinado = df_combinado[~df_combinado.index.duplicated(keep='last')]
    
    # Ordenar por fecha
    df_combinado = df_combinado.sort_index()
    
    # Rellenar valores nulos con métodos más avanzados
    for col in columnas_comunes[1:]:
        if df_combinado[col].isnull().any():
            # Usar interpolación específica para cada variable
            if col in ['temperatura_C', 'humedad_relativa']:
                # Para temp y humedad, usar interpolación tiempo con límites
                df_combinado[col] = df_combinado[col].interpolate(method='time', limit_direction='both')
            elif col == 'precipitacion_mm':
                # Para precipitación, usar ffill pero con límite (la lluvia no dura para siempre)
                df_combinado[col] = df_combinado[col].fillna(0)  # Asumir 0 precipitación por defecto
            else:
                # Para otras variables, interpolación linear simple
                df_combinado[col] = df_combinado[col].interpolate(method='linear', limit_direction='both')
    
    # MODIFICACIÓN CLAVE: Guardar dataset combinado preservando la columna fecha
    # Primero resetear el índice para convertirlo en columna
    df_output = df_combinado.reset_index()
    
    # Asegurarse de que la columna se llame 'fecha'
    if 'index' in df_output.columns and 'fecha' not in df_output.columns:
        df_output = df_output.rename(columns={'index': 'fecha'})
        
    # Guardar el dataset con la columna fecha explícita
    df_output.to_csv(ruta_salida, index=False)
    
    # Información detallada del dataset combinado
    fechas_unicas = pd.Series(df_combinado.index.date).unique()
    print(f"Dataset combinado creado exitosamente con {len(df_combinado)} registros")
    print(f"Rango de fechas: {df_combinado.index.min()} hasta {df_combinado.index.max()}")
    print(f"Días únicos: {len(fechas_unicas)}")
    total_dias_posibles = (df_combinado.index.max() - df_combinado.index.min()).days + 1
    huecos = total_dias_posibles - len(fechas_unicas)
    print(f"Huecos en la serie temporal: {huecos} días")
    
    return df_combinado
# Fin del metodo que de Dataset de 7 años

# Obtener directorio actual y directorio del script
directorio_actual = os.getcwd()
directorio_script = os.path.dirname(os.path.abspath(__file__))

print(f"Directorio de ejecución: {directorio_actual}")
print(f"Directorio del script: {directorio_script}")

# Cambiar al directorio del script para que las rutas relativas funcionen
os.chdir(directorio_script)
print(f"Cambiado al directorio: {os.getcwd()}")

# Configuración para reducir los mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VentanaProgreso(tk.Toplevel):
    def __init__(self, parent, titulo="Procesando"):
        super().__init__(parent)
        self.title(titulo)
        self.geometry("300x150")
        self.resizable(False, False)
        
        # Hacer la ventana modal
        self.transient(parent)
        self.grab_set()
        
        # Centrar la ventana
        self.center_window()
        
        # Frame principal
        self.frame = ttk.Frame(self)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables
        self.progress_var = tk.DoubleVar()
        
        # Widgets
        self.label = ttk.Label(self.frame, text="0%", font=('Arial', 10, 'bold'))
        self.label.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            self.frame, 
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=200
        )
        self.progress.pack(pady=10)
        
        self.message = ttk.Label(self.frame, text="", font=('Arial', 9))
        self.message.pack(pady=10)
        
        # Protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def on_closing(self):
        """Maneja el intento de cerrar la ventana"""
        # Simplemente ignora el intento de cerrar
        pass
    
    def update_progress(self, value, message=""):
        """Actualiza el progreso de forma segura"""
        try:
            if self.winfo_exists():
                self.progress_var.set(value)
                self.label.config(text=f"{value:.1f}%")
                if message:
                    self.message.config(text=message)
                self.update_idletasks()
        except tk.TclError:
            pass  # Ignorar errores si la ventana ya no existe
    
    def safe_destroy(self):
        """Destruye la ventana de forma segura"""
        try:
            if self.winfo_exists():
                self.destroy()
        except tk.TclError:
            pass  # Ignorar errores si la ventana ya no existe
    
    def on_closing(self):
        """Maneja el intento de cerrar la ventana"""
        # Simplemente ignora el intento de cerrar
        pass
    
    def update_progress(self, value, message=""):
        """Actualiza el progreso de forma segura"""
        try:
            if self.winfo_exists():
                self.progress_var.set(value)
                self.label.config(text=f"{value:.1f}%")
                if message:
                    self.message.config(text=message)
                self.update_idletasks()
        except tk.TclError:
            pass  # Ignorar errores si la ventana ya no existe
    
    def safe_destroy(self):
        """Destruye la ventana de forma segura"""
        try:
            if self.winfo_exists():
                self.destroy()
        except tk.TclError:
            pass
class VentanaVisualizacion(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Pronóstico de Temperatura y Confianza")
        self.geometry("1000x600")
        self.minsize(800, 400)
        
        # Centrar la ventana
        self.center_window()
        
        # Crear frames principales
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pred_frame = ttk.Frame(self)
        self.pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar barra de herramientas
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Botones de control
        self.create_toolbar()
        
        # Configurar eventos de ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables de estado
        self.current_view = "temperature"  # Valor por defecto
        
        # Frame para barras de navegación
        self.nav_frame = ttk.Frame(self)
        self.nav_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    def actualizar_grafica(self, predicciones, visualizador):
        """Actualiza la gráfica de la ventana con nuevas predicciones"""
        try:
            # Limpiar main_frame
            for widget in self.main_frame.winfo_children():
                widget.destroy()
                
            # Crear gráfica de temperatura usando el visualizador
            fig = visualizador.crear_grafica_temperatura(predicciones)
            
            # Crear canvas
            canvas = FigureCanvasTkAgg(fig, self.main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Crear barra de navegación
            if hasattr(self, 'pred_frame') and self.pred_frame:
                for widget in self.pred_frame.winfo_children():
                    widget.destroy()
            else:
                self.pred_frame = ttk.Frame(self.main_frame)
                self.pred_frame.pack(fill=tk.X)
                
            toolbar = NavigationToolbar2Tk(canvas, self.pred_frame)
            toolbar.update()
            
            # Traer ventana al frente
            self.lift()
            
        except Exception as e:
            import traceback
            print(f"Error al actualizar gráfica de temperatura: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Error al actualizar gráfica: {str(e)}")
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        # Añadir este método a la clase VentanaVisualizacion

    def create_toolbar(self):
        """Crea la barra de herramientas con controles"""
        # Botón para alternar vista
        self.toggle_btn = ttk.Button(
            self.toolbar_frame,
            text="Cambiar Vista",
            command=self.toggle_view
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para exportar
        self.export_btn = ttk.Button(
            self.toolbar_frame,
            text="Exportar Gráfica",
            command=self.export_graph
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para actualizar
        self.refresh_btn = ttk.Button(
            self.toolbar_frame,
            text="Actualizar",
            command=self.refresh_view
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

    def verificar_visibilidad(self):
        """Verifica si todos los elementos son visibles y ajusta si es necesario"""
        try:
            # Forzar actualización para obtener dimensiones correctas
            self.update_idletasks()
            
            # Obtener dimensiones de la ventana
            window_width = self.winfo_width()
            window_height = self.winfo_height()
            
            # Definir dimensiones mínimas recomendadas basadas en el contenido
            min_width = 1000
            min_height = 700
            
            # Verificar si es necesario ajustar
            if window_width < min_width or window_height < min_height:
                # Calcular nuevas dimensiones
                new_width = max(window_width, min_width)
                new_height = max(window_height, min_height)
                
                # Limitar al tamaño de pantalla
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                new_width = min(new_width, screen_width * 0.9)
                new_height = min(new_height, screen_height * 0.9)
                
                # Aplicar nuevo tamaño
                self.geometry(f"{int(new_width)}x{int(new_height)}")
                
                # Centrar en pantalla
                x = (screen_width - new_width) // 2
                y = (screen_height - new_height) // 2
                self.geometry(f"{int(new_width)}x{int(new_height)}+{x}+{y}")
                
                print(f"Ajustado tamaño a {new_width}x{new_height}")
                
        except Exception as e:
            print(f"Error al verificar visibilidad: {e}")

    def add_scrollbars(self):
        """Añade barras de desplazamiento si el contenido no cabe"""
        try:
            # Verificar si ya tenemos canvas
            if hasattr(self, 'canvas'):
                return
                
            # Guardar referencia al contenido original
            original_content = self.main_frame
            
            # Crear nuevo canvas con scrollbars
            self.canvas = tk.Canvas(self.container)
            self.scrollbar_y = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
            self.scrollbar_x = ttk.Scrollbar(self.container, orient="horizontal", command=self.canvas.xview)
            
            # Configurar canvas
            self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
            
            # Ubicación de widgets
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.scrollbar_y.grid(row=0, column=1, sticky="ns")
            self.scrollbar_x.grid(row=1, column=0, sticky="ew")
            
            # Configurar grid
            self.container.grid_rowconfigure(0, weight=1)
            self.container.grid_columnconfigure(0, weight=1)
            
            # Crear nuevo frame dentro del canvas
            self.scrollable_frame = ttk.Frame(self.canvas)
            self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            
            # Mover contenido original al nuevo frame
            original_content.pack_forget()
            original_content.pack(in_=self.scrollable_frame, fill=tk.BOTH, expand=True)
            
            # Configurar eventos de actualización
            self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            
        except Exception as e:
            print(f"Error al añadir scrollbars: {e}")

    def _on_frame_configure(self, event=None):
        """Actualiza scroll region cuando el frame cambia de tamaño"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Ajusta el ancho del frame interno cuando el canvas cambia de tamaño"""
        if hasattr(self, 'canvas_frame') and hasattr(self, 'scrollable_frame'):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_frame, width=canvas_width)
    def mostrar_figura(self, fig):
        """Muestra una figura ya generada en la ventana"""
        # Limpiar main_frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Crear canvas
        canvas = FigureCanvasTkAgg(fig, self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear barra de navegación
        if hasattr(self, 'pred_frame') and self.pred_frame:
            for widget in self.pred_frame.winfo_children():
                widget.destroy()
        else:
            self.pred_frame = ttk.Frame(self.main_frame)
            self.pred_frame.pack(fill=tk.X)
            
        toolbar = NavigationToolbar2Tk(canvas, self.pred_frame)
        toolbar.update()

    def create_toolbar(self):
        """Crea la barra de herramientas con controles"""
        # Botón para alternar vista
        self.toggle_btn = ttk.Button(
            self.toolbar_frame,
            text="Cambiar Vista",
            command=self.toggle_view
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para exportar
        self.export_btn = ttk.Button(
            self.toolbar_frame,
            text="Exportar Gráfica",
            command=self.export_graph
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para actualizar
        self.refresh_btn = ttk.Button(
            self.toolbar_frame,
            text="Actualizar",
            command=self.refresh_view
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
    def toggle_view(self):
        """Alterna entre diferentes vistas de visualización"""
        self.current_view = "confidence" if self.current_view == "temperature" else "temperature"
        self.refresh_view()
        
    def export_graph(self):
        """Exporta la gráfica actual como imagen"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                for widget in self.main_frame.winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        widget.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Éxito", "Gráfica exportada exitosamente")
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar la gráfica: {str(e)}")
            
    def refresh_view(self):
        """Actualiza completamente la visualización del pronóstico"""
        try:
            # Mostrar ventana de progreso
            progreso = VentanaProgreso(self, "Actualizando Pronóstico")
            progreso.update_progress(10, "Preparando datos...")
            
            def update_task():
                try:
                    # Guardar las selecciones actuales del usuario
                    selecciones_usuario = {}
                    for fecha_periodo, widget_info in self.feedback_widgets.items():
                        combo = widget_info.get('combo')
                        if combo:
                            selecciones_usuario[fecha_periodo] = combo.get()
                    
                    progreso.update_progress(30, "Generando predicciones...")
                    
                    # Obtener nuevas predicciones
                    if hasattr(self, '_visualizador') and self._visualizador and hasattr(self._visualizador, 'predictor'):
                        predictor = self._visualizador.predictor
                        
                        # CORRECCIÓN: Verificar si hay un dataset disponible
                        if not hasattr(predictor, 'dataset') or predictor.dataset is None:
                            print("No hay dataset disponible para hacer predicciones. Cargando modelo...")
                            
                            # Intenta cargar el modelo para asegurar que los datos estén disponibles
                            try:
                                progreso.update_progress(40, "Cargando modelo...")
                                predictor.cargar_modelo_guardado()
                                print("Modelo cargado correctamente")
                            except Exception as load_err:
                                print(f"Error al cargar modelo: {load_err}")
                            
                            # Verificar nuevamente el dataset
                            if not hasattr(predictor, 'dataset') or predictor.dataset is None:
                                print("Creando dataset mínimo para predicciones...")
                                
                                # Crear un dataset mínimo con datos de la última retroalimentación
                                from datetime import datetime, timedelta
                                import pandas as pd
                                import numpy as np
                                
                                # Obtener hora actual
                                now = datetime.now()
                                
                                # Crear DataFrame de ejemplo con datos mínimos
                                datos = []
                                for i in range(24):  # 24 horas
                                    fecha = now - timedelta(hours=i)
                                    datos.append({
                                        'fecha': fecha,
                                        'temperatura_C': 15.0 + np.random.uniform(-3, 3),
                                        'humedad_relativa': 70.0 + np.random.uniform(-10, 10),
                                        'precipitacion_mm': 0.0,
                                        'cobertura_nubes_octas': 4.0,
                                        'velocidad_viento_kmh': 5.0,
                                        'radiacion_solar_J_m2': 5000.0
                                    })
                                
                                # Crear DataFrame temporal
                                df_temp = pd.DataFrame(datos)
                                df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
                                df_temp.set_index('fecha', inplace=True)
                                
                                # Asignar al predictor
                                predictor.dataset = df_temp
                                print(f"Dataset mínimo creado con {len(df_temp)} registros")
                        
                        progreso.update_progress(60, "Aplicando modelo...")
                        
                        # Ahora intentamos las predicciones con el dataset asegurado
                        try:
                            predicciones = predictor.predecir_proximo_periodo(predictor.dataset)
                            
                            progreso.update_progress(80, "Actualizando visualización...")
                            
                            # Actualiza la UI en el hilo principal
                            self.after(0, lambda: self._actualizar_ui(predicciones, selecciones_usuario))
                            
                        except Exception as pred_err:
                            print(f"Error en predicción: {pred_err}")
                            self.after(0, lambda: messagebox.showerror("Error", 
                                                    "No se pudieron actualizar las predicciones.\n\nPor favor, intenta con el botón 'Actualizar'."))
                    else:
                        print("No se pudo acceder al visualizador o predictor")
                        self.after(0, lambda: messagebox.showinfo("Información", "No se pudo acceder al motor de predicción. Intenta con el botón 'Actualizar'."))
                    
                    progreso.update_progress(100, "¡Visualización actualizada!")
                    self.after(500, progreso.safe_destroy)
                        
                except Exception as e:
                    print(f"Error al refrescar vista: {e}")
                    import traceback
                    traceback.print_exc()
                    self.after(0, lambda: messagebox.showerror("Error", "No se pudo actualizar la visualización."))
                    progreso.safe_destroy()
            
            # Ejecutar en segundo plano
            threading.Thread(target=update_task, daemon=True).start()
            
        except Exception as e:
            print(f"Error al iniciar actualización: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", "No se pudo iniciar la actualización.")
            if 'progreso' in locals():
                progreso.safe_destroy()
        
    def _actualizar_ui(self, predicciones, selecciones_usuario):
        """Método auxiliar para actualizar la UI desde el hilo principal"""
        # Limpiar la vista actual
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Recrear toda la visualización con las nuevas predicciones
        self.actualizar_grafica(predicciones, self._visualizador)
        
        # Restaurar las selecciones del usuario
        for fecha_periodo, categoria in selecciones_usuario.items():
            if fecha_periodo in self.feedback_widgets:
                combo = self.feedback_widgets[fecha_periodo].get('combo')
                if combo:
                    try:
                        combo.set(categoria)
                    except:
                        pass
        
        # Mensaje de confirmación discreto
        confirmacion = ttk.Label(self.main_frame, 
                            text="Visualización actualizada con tus observaciones", 
                            font=('Arial', 10, 'italic'),
                            foreground='green')
        confirmacion.place(relx=0.5, rely=0.98, anchor='center')
        
        # Eliminar el mensaje después de unos segundos
        self.after(5000, lambda: confirmacion.destroy() if confirmacion.winfo_exists() else None)
    def actualizar_graficas_iniciales(self, figuras):
        """Actualiza las gráficas iniciales"""
        try:
            # Limpiar frames
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            for widget in self.pred_frame.winfo_children():
                widget.destroy()
                
            # Mostrar gráfica de series temporales
            if 'series_temporal' in figuras:
                canvas_series = FigureCanvasTkAgg(figuras['series_temporal'], self.main_frame)
                canvas_series.draw()
                canvas_series.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Agregar barra de navegación
                toolbar_series = NavigationToolbar2Tk(canvas_series, self.pred_frame)
                toolbar_series.update()
                
            # Mostrar gráfica de distribución
            if 'distribucion' in figuras:
                canvas_dist = FigureCanvasTkAgg(figuras['distribucion'], self.main_frame)
                canvas_dist.draw()
                canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Agregar barra de navegación
                toolbar_dist = NavigationToolbar2Tk(canvas_dist, self.pred_frame)
                toolbar_dist.update()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al actualizar las gráficas iniciales: {str(e)}")
            
    def on_closing(self):
        """Maneja el cierre de la ventana"""
        self.destroy()

class VentanaPronosticoDetallado(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Pronóstico Detallado por Períodos")
        self.geometry("800x600")
        self.minsize(600, 400)
        # Diccionario de emojis de respaldo
        self.emoji_respaldo = {
            'Madrugada': '🌙',
            'Mañana': '🌄',
            'Tarde': '☀️',
            'Noche': '🌠',
            'Frío': '❄️',
            'Templado': '🌤️',
            'Cálido': '🔥',
            'Lluvia': '🌧️',
            'Llovizna': '🌦️',
            'Nublado': '☁️',
            'Soleado': '☀️'
        }
        # Centrar la ventana
        self.center_window()
        
        # IMPORTANTE: Crear frame contenedor principal
        self.container = ttk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # PRIMERO: Crear frame de controles en la parte superior
        self.controls_frame = ttk.Frame(self.container)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)
        
        # DESPUÉS: Crear frame para el contenido principal
        self.main_frame = ttk.Frame(self.container)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Inicializar variables importantes
        self._visualizador = None
        self.feedback_widgets = {}
        self.tk_images = []
        
        # Crear controles básicos inmediatamente
        self.create_basic_controls()
        
        # Configurar eventos
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Binding para redimensionamiento con corrección
        self.bind("<Configure>", self._on_configure)
    def verificar_visibilidad(self):
        """Verifica si todos los elementos son visibles y ajusta si es necesario"""
        try:
            # Obtener dimensiones de la ventana
            window_width = self.winfo_width()
            window_height = self.winfo_height()
            
            # Obtener dimensiones del contenido
            content_width = 0
            content_height = 0
            
            for widget in self.main_frame.winfo_children():
                widget_width = widget.winfo_width() + widget.winfo_x()
                widget_height = widget.winfo_height() + widget.winfo_y()
                content_width = max(content_width, widget_width)
                content_height = max(content_height, widget_height)
            
            # Verificar si es necesario ajustar
            if content_width > window_width or content_height > window_height:
                # Calcular nuevas dimensiones con un margen
                new_width = max(window_width, content_width + 20)
                new_height = max(window_height, content_height + 20)
                
                # Limitamos el tamaño máximo a la pantalla
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                new_width = min(new_width, screen_width * 0.9)
                new_height = min(new_height, screen_height * 0.9)
                
                # Aplicar nuevo tamaño
                self.geometry(f"{int(new_width)}x{int(new_height)}")
                
                # Si el contenido es más grande que la pantalla, añadir scrollbars
                if content_width > screen_width * 0.9 or content_height > screen_height * 0.9:
                    self.add_scrollbars()
        except Exception as e:
            print(f"Error al verificar visibilidad: {e}")

    def add_scrollbars(self):
        """Añade barras de desplazamiento si el contenido no cabe"""
        try:
            # Verificar si ya tenemos canvas
            if hasattr(self, 'canvas'):
                return
                
            # Guardar referencia al contenido original
            original_content = self.main_frame
            
            # Crear nuevo canvas con scrollbars
            self.canvas = tk.Canvas(self.container)
            self.scrollbar_y = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
            self.scrollbar_x = ttk.Scrollbar(self.container, orient="horizontal", command=self.canvas.xview)
            
            # Configurar canvas
            self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
            
            # Ubicación de widgets
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.scrollbar_y.grid(row=0, column=1, sticky="ns")
            self.scrollbar_x.grid(row=1, column=0, sticky="ew")
            
            # Configurar grid
            self.container.grid_rowconfigure(0, weight=1)
            self.container.grid_columnconfigure(0, weight=1)
            
            # Crear nuevo frame dentro del canvas
            self.scrollable_frame = ttk.Frame(self.canvas)
            self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            
            # Mover contenido original al nuevo frame
            original_content.pack_forget()
            original_content.pack(in_=self.scrollable_frame, fill=tk.BOTH, expand=True)
            
            # Configurar eventos de actualización
            self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            
        except Exception as e:
            print(f"Error al añadir scrollbars: {e}")

    def _on_frame_configure(self, event=None):
        """Actualiza scroll region cuando el frame cambia de tamaño"""
        if hasattr(self, 'canvas'):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Ajusta el ancho del frame interno cuando el canvas cambia de tamaño"""
        if hasattr(self, 'canvas_frame') and hasattr(self, 'scrollable_frame'):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_frame, width=canvas_width)

    def auto_adjust_window(self):
        """Ajusta automáticamente el tamaño de la ventana al contenido"""
        try:
            # Forzar actualización para que las medidas sean correctas
            self.update_idletasks()
            
            # Verificar visibilidad y ajustar
            self.verificar_visibilidad()
            
            # Centrar ventana
            self.center_window()
            
            # Mostrar confirmación
            messagebox.showinfo(
                "Ajuste Automático", 
                "La ventana ha sido ajustada automáticamente para mostrar todo el contenido."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error al ajustar ventana: {str(e)}")
    def auto_adjust_window(self):
            """Ajusta automáticamente el tamaño de la ventana al contenido"""
            try:
                # Forzar actualización para que las medidas sean correctas
                self.update_idletasks()
                
                # Verificar visibilidad y ajustar
                self.verificar_visibilidad()
                
                # Centrar ventana
                self.center_window()
                
                # Mensaje de confirmación
                messagebox.showinfo(
                    "Ajuste Automático", 
                    "La ventana ha sido ajustada automáticamente para mostrar todo el contenido."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error al ajustar ventana: {str(e)}")
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    def guardar_retroalimentacion(self, fecha, periodo, cat_original, cat_usuario):
        """Guarda la retroalimentación del usuario para uso futuro"""
        try:
            # Crear directorio si no existe
            feedback_dir = "datos_retroalimentacion"
            if not os.path.exists(feedback_dir):
                os.makedirs(feedback_dir)
                
            # Nombre de archivo para el registro de retroalimentación
            feedback_file = os.path.join(feedback_dir, "feedback_usuario.csv")
            
            # Crear dataframe para el nuevo feedback
            nueva_entrada = pd.DataFrame({
                'fecha': [fecha.strftime('%Y-%m-%d')],
                'periodo': [periodo],
                'hora': [self.obtener_hora_representativa(periodo)],
                'categoria_original': [cat_original],
                'categoria_usuario': [cat_usuario],
                'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            
            # Verificar si el archivo existe y cargar datos anteriores
            if os.path.exists(feedback_file):
                df_existente = pd.read_csv(feedback_file)
                df_combinado = pd.concat([df_existente, nueva_entrada])
            else:
                df_combinado = nueva_entrada
                
            # Guardar el registro actualizado
            df_combinado.to_csv(feedback_file, index=False)
            
            print(f"Retroalimentación guardada en {feedback_file}")
            
        except Exception as e:
            print(f"Error al guardar retroalimentación: {str(e)}")
    def obtener_hora_representativa(self, periodo):
        """Devuelve la hora representativa para cada período"""
        horas_representativas = {
            'Madrugada': '01:00',
            'Mañana': '07:00',
            'Tarde': '13:00',
            'Noche': '19:00'
        }
        return horas_representativas.get(periodo, '00:00')
    def actualizar_modelo_con_retroalimentacion(self, fecha, periodo, nueva_categoria, peso_observacion=10.0):
        """Actualiza el modelo con la retroalimentación del usuario"""
        try:
            # Verificar si tenemos acceso al predictor
            if not hasattr(self, '_visualizador') or not hasattr(self._visualizador, 'predictor'):
                messagebox.showinfo("Información", "No se puede actualizar el modelo en este momento")
                return
                    
            predictor = self._visualizador.predictor
                
            # Crear registro de datos para actualizar modelo
            hora = self.obtener_hora_representativa(periodo)
            fecha_hora = datetime.combine(fecha, datetime.strptime(hora, '%H:%M').time())
                
            # Ventana de progreso
            progress = VentanaProgreso(self, "Actualizando modelo")
            progress.update_progress(10, "Preparando datos...")
                
            def actualizar_modelo_thread():
                try:
                    # Crear un conjunto mínimo de datos
                    try:
                        temperatura = predictor.predecir_temperatura(None, 0)
                    except Exception as temp_err:
                        print(f"Error al predecir temperatura: {temp_err}")
                        temperatura = 15.0  # Valor por defecto
                    
                    # Crear DataFrame con más datos de contexto para evitar filtrado excesivo
                    datos_adicionales = []
                    
                    # Agregar el dato principal de retroalimentación
                    datos_feedback = pd.DataFrame({
                        'fecha': [fecha_hora],
                        'temperatura_C': [temperatura],
                        'humedad_relativa': [70.0],
                        'precipitacion_mm': [0.0 if 'Lluvia' not in nueva_categoria else 0.8],
                        'cobertura_nubes_octas': [7.0 if 'Nublado' in nueva_categoria else 3.0],
                        'velocidad_viento_kmh': [5.0],
                        'radiacion_solar_J_m2': [10000.0 if 'Soleado' in nueva_categoria else 5000.0],
                        'categoria_clima': [nueva_categoria],
                        'verificado': True  # CORREGIDO: Usamos un valor booleano, no una lista
                    })
                    
                    # Agregar algunos datos sintéticos adicionales para evitar problemas con filtrado
                    # Agregamos datos para las últimas 24 horas con pequeñas variaciones
                    for hora_offset in range(1, 25):
                        hora_adicional = fecha_hora - timedelta(hours=hora_offset)
                        # Variación aleatoria pequeña para los valores
                        temp_var = temperatura + np.random.uniform(-1.0, 1.0)
                        hum_var = 70.0 + np.random.uniform(-5.0, 5.0)
                        datos_adicionales.append({
                            'fecha': hora_adicional,
                            'temperatura_C': temp_var,
                            'humedad_relativa': hum_var,
                            'precipitacion_mm': 0.0 if 'Lluvia' not in nueva_categoria else 0.3,
                            'cobertura_nubes_octas': 6.0 if 'Nublado' in nueva_categoria else 2.0,
                            'velocidad_viento_kmh': 5.0 + np.random.uniform(-1.0, 1.0),
                            'radiacion_solar_J_m2': 8000.0 if 'Soleado' in nueva_categoria else 4000.0,
                            'categoria_clima': nueva_categoria,
                            'verificado': False  # CORREGIDO: Usamos un valor booleano, no una lista
                        })
                    
                    # Crear DataFrame con todos los datos
                    datos_extra = pd.DataFrame(datos_adicionales)
                    datos_completos = pd.concat([datos_feedback, datos_extra], ignore_index=True)
                        
                    # Establecer fecha como índice
                    datos_completos['fecha'] = pd.to_datetime(datos_completos['fecha'])
                    datos_completos.set_index('fecha', inplace=True)
                        
                    progress.update_progress(40, "Actualizando modelo...")
                    
                    # Indicar que estos son datos de retroalimentación para evitar filtrado
                    predictor._omitir_filtrado_temporal = True
                        
                    # Función de peso que maneja correctamente los valores y aplica MAYOR PESO
                    def sample_weight_function(df):
                        # Asegurarse de que verificado sea una serie de booleanos
                        verificado = df['verificado']
                        if isinstance(verificado, pd.Series):
                            # Convertir a numpy array para usar with np.where
                            verificado_array = verificado.astype(bool).values
                        else:
                            # Si es un solo valor, crear array del tamaño adecuado
                            verificado_array = np.array([bool(verificado)] * len(df))
                        
                        # Asignar pesos - USAR PESO_OBSERVACION PASADO COMO PARÁMETRO
                        return np.where(verificado_array, peso_observacion, 1.0)
                        
                    # Actualizar el modelo con estos datos
                    predictor.actualizar_modelo_con_nuevos_datos(
                        datos_completos, 
                        guardar=True, 
                        sample_weights=sample_weight_function  # Usar nuestra función más robusta
                    )
                    
                    # Restaurar configuración normal
                    predictor._omitir_filtrado_temporal = False
                        
                    progress.update_progress(90, "Finalizado...")
                        
                    # ELIMINADO: No mostrar mensaje aquí, se mostrará después de refrescar la vista
                    
                    # Cerrar ventana de progreso
                    progress.safe_destroy()
                        
                except Exception as update_err:
                    # Capturar el mensaje de error específicamente
                    error_msg = str(update_err)
                    import traceback
                    print(f"Error detallado: {traceback.format_exc()}")  # Añadir trazabilidad detallada
                    progress.safe_destroy()
                    self.after(100, lambda: messagebox.showerror(
                        "Error", 
                        f"Error al actualizar el modelo: {error_msg}"
                    ))
                
            # Ejecutar actualización en segundo plano
            threading.Thread(target=actualizar_modelo_thread, daemon=True).start()
                
        except Exception as e:
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")  # Añadir trazabilidad detallada
            messagebox.showerror("Error", f"Error al iniciar actualización: {str(e)}")

    def mostrar_historial_retroalimentacion(self):
        """Muestra el historial de retroalimentaciones del usuario"""
        try:
            feedback_file = os.path.join("datos_retroalimentacion", "feedback_usuario.csv")
            
            if not os.path.exists(feedback_file):
                messagebox.showinfo("Información", "No hay retroalimentaciones registradas aún")
                return
                
            # Cargar datos
            df_feedback = pd.read_csv(feedback_file)
            
            # Crear ventana para mostrar datos
            ventana = tk.Toplevel(self)
            ventana.title("Historial de Retroalimentación")
            ventana.geometry("800x400")
            
            # Crear tabla
            frame = ttk.Frame(ventana)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Crear Treeview
            tree = ttk.Treeview(frame)
            tree["columns"] = ("fecha", "periodo", "original", "usuario", "timestamp")
            
            # Configurar columnas
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("fecha", width=100, anchor=tk.W)
            tree.column("periodo", width=80, anchor=tk.W)
            tree.column("original", width=150, anchor=tk.W)
            tree.column("usuario", width=150, anchor=tk.W)
            tree.column("timestamp", width=150, anchor=tk.W)
            
            # Configurar encabezados
            tree.heading("#0", text="", anchor=tk.W)
            tree.heading("fecha", text="Fecha", anchor=tk.W)
            tree.heading("periodo", text="Período", anchor=tk.W)
            tree.heading("original", text="Predicción Original", anchor=tk.W)
            tree.heading("usuario", text="Corrección Usuario", anchor=tk.W)
            tree.heading("timestamp", text="Momento", anchor=tk.W)
            
            # Insertar datos
            for idx, row in df_feedback.iterrows():
                tree.insert("", idx, values=(
                    row['fecha'],
                    row['periodo'],
                    row['categoria_original'],
                    row['categoria_usuario'],
                    row['timestamp']
                ))
            
            # Agregar scrollbar
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar historial: {str(e)}")
    def create_basic_controls(self):
        """Crea controles básicos con énfasis visual"""
        # Asegurarse de que el frame de controles existe
        if not hasattr(self, 'controls_frame') or self.controls_frame is None:
            self.controls_frame = ttk.Frame(self)
            self.controls_frame.pack(fill=tk.X, side=tk.TOP, pady=5, padx=5)
        
        # Limpiar controles existentes
        for widget in self.controls_frame.winfo_children():
            widget.destroy()
        
        # Estilo para botones más visibles
        style = ttk.Style()
        style.configure('Control.TButton', 
                    font=('Arial', 10, 'bold'),
                    padding=5)
        
        # Frame para espaciado y visual
        separator = ttk.Frame(self.controls_frame, height=2)
        separator.pack(fill=tk.X, pady=2)
        
        # Botón de actualización con mejor visibilidad
        update_btn = ttk.Button(
            self.controls_frame,
            text="Actualizar Pronóstico",
            command=self.refresh_view,
            style='Control.TButton',
            width=20
        )
        update_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón de exportación con mejor visibilidad
        export_btn = ttk.Button(
            self.controls_frame,
            text="Exportar Pronóstico",
            command=self.export_forecast,
            style='Control.TButton',
            width=20
        )
        export_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón para historial
        history_btn = ttk.Button(
            self.controls_frame,
            text="Ver Historial",
            command=self.mostrar_historial_retroalimentacion,
            style='Control.TButton',
            width=20
        )
        history_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón para ajuste automático
        adjust_btn = ttk.Button(
            self.controls_frame,
            text="Ajustar a Ventana",
            command=self.auto_adjust_window,
            style='Control.TButton',
            width=20
        )
        adjust_btn.pack(side=tk.LEFT, padx=10)
        
        # Frame para espaciado y visual (separador)
        separator2 = ttk.Frame(self.controls_frame, height=2, style='TSeparator')
        separator2.pack(fill=tk.X, pady=5)
    
    def refresh_view(self):
        """Actualiza completamente la visualización del pronóstico"""
        try:
            # Guardar las selecciones actuales del usuario
            selecciones_usuario = {}
            for fecha_periodo, widget_info in self.feedback_widgets.items():
                combo = widget_info.get('combo')
                if combo:
                    selecciones_usuario[fecha_periodo] = combo.get()
            
            # Obtener nuevas predicciones
            if hasattr(self, '_visualizador') and self._visualizador and hasattr(self._visualizador, 'predictor'):
                predictor = self._visualizador.predictor
                
                # CORRECCIÓN: Verificar si hay un dataset disponible
                if not hasattr(predictor, 'dataset') or predictor.dataset is None:
                    print("No hay dataset disponible para hacer predicciones. Cargando modelo...")
                    
                    # Intenta cargar el modelo para asegurar que los datos estén disponibles
                    try:
                        predictor.cargar_modelo_guardado()
                        print("Modelo cargado correctamente")
                    except Exception as load_err:
                        print(f"Error al cargar modelo: {load_err}")
                    
                    # Verificar nuevamente el dataset
                    if not hasattr(predictor, 'dataset') or predictor.dataset is None:
                        print("Creando dataset mínimo para predicciones...")
                        
                        # Crear un dataset mínimo con datos de la última retroalimentación
                        from datetime import datetime, timedelta
                        import pandas as pd
                        import numpy as np
                        
                        # Obtener hora actual
                        now = datetime.now()
                        
                        # Crear DataFrame de ejemplo con datos mínimos
                        datos = []
                        for i in range(24):  # 24 horas
                            fecha = now - timedelta(hours=i)
                            datos.append({
                                'fecha': fecha,
                                'temperatura_C': 15.0 + np.random.uniform(-3, 3),
                                'humedad_relativa': 70.0 + np.random.uniform(-10, 10),
                                'precipitacion_mm': 0.0,
                                'cobertura_nubes_octas': 4.0,
                                'velocidad_viento_kmh': 5.0,
                                'radiacion_solar_J_m2': 5000.0
                            })
                        
                        # Crear DataFrame temporal
                        df_temp = pd.DataFrame(datos)
                        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
                        df_temp.set_index('fecha', inplace=True)
                        
                        # Asignar al predictor
                        predictor.dataset = df_temp
                        print(f"Dataset mínimo creado con {len(df_temp)} registros")
                
                # Ahora intentamos las predicciones con el dataset asegurado
                try:
                    predicciones = predictor.predecir_proximo_periodo(predictor.dataset)
                    
                    # Limpiar la vista actual
                    for widget in self.main_frame.winfo_children():
                        widget.destroy()
                    
                    # Recrear toda la visualización con las nuevas predicciones
                    self.actualizar_grafica(predicciones, self._visualizador)
                    
                    # Restaurar las selecciones del usuario
                    for fecha_periodo, categoria in selecciones_usuario.items():
                        if fecha_periodo in self.feedback_widgets:
                            combo = self.feedback_widgets[fecha_periodo].get('combo')
                            if combo:
                                try:
                                    combo.set(categoria)
                                except:
                                    pass
                    
                    # Mensaje de confirmación discreto
                    confirmacion = ttk.Label(self.main_frame, 
                                        text="Visualización actualizada con tus observaciones", 
                                        font=('Arial', 10, 'italic'),
                                        foreground='green')
                    confirmacion.place(relx=0.5, rely=0.98, anchor='center')
                    
                    # Eliminar el mensaje después de unos segundos
                    self.after(5000, lambda: confirmacion.destroy() if confirmacion.winfo_exists() else None)
                except Exception as pred_err:
                    print(f"Error en predicción: {pred_err}")
                    messagebox.showerror("Error", 
                                        f"No se pudieron actualizar las predicciones.\n\nPor favor, intenta con el botón 'Actualizar'.")
            else:
                print("No se pudo acceder al visualizador o predictor")
                messagebox.showinfo("Información", "No se pudo acceder al motor de predicción. Intenta con el botón 'Actualizar'.")
                
        except Exception as e:
            print(f"Error al refrescar vista: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", "No se pudo actualizar la visualización.")
        
    def export_forecast(self):
        """Exporta la visualización actual como imagen del frame principal"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if file_path:
                # Esperar a que la interfaz se actualice
                self.update_idletasks()
                
                # Capturar solo el frame principal, no toda la ventana
                from PIL import ImageGrab
                
                # Coordenadas del frame principal dentro de la pantalla
                x = self.main_frame.winfo_rootx()
                y = self.main_frame.winfo_rooty()
                width = self.main_frame.winfo_width()
                height = self.main_frame.winfo_height()
                
                print(f"Capturando solo el frame principal: x={x}, y={y}, ancho={width}, alto={height}")
                
                # Capturar el área específica
                img = ImageGrab.grab((x, y, x+width, y+height))
                img.save(file_path)
                
                messagebox.showinfo("Éxito", "Pronóstico exportado exitosamente")
        except Exception as e:
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Error al exportar el pronóstico: {str(e)}")
    
    def on_closing(self):
        """Maneja el cierre de la ventana"""
        self.destroy()
    def verificar_directorio_imagenes(self):
        """Verifica que el directorio de imágenes exista y es accesible"""
        try:
            # Obtener directorio del script actual
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Posibles rutas para el directorio de imágenes
            posibles_rutas = [
                'Imagenes-Clima',
                os.path.join(script_dir, 'Imagenes-Clima'),
                os.path.join(os.getcwd(), 'Imagenes-Clima')
            ]
            
            directorio_encontrado = None
            for ruta in posibles_rutas:
                if os.path.exists(ruta) and os.path.isdir(ruta):
                    directorio_encontrado = ruta
                    print(f"Directorio de imágenes encontrado: {ruta}")
                    break
                    
            if not directorio_encontrado:
                print("ADVERTENCIA: No se encontró el directorio de imágenes")
                print("Buscando en rutas alternativas...")
                
                # Buscar uno o dos niveles hacia arriba
                parent_dir = os.path.dirname(script_dir)
                grandparent_dir = os.path.dirname(parent_dir)
                
                otras_rutas = [
                    os.path.join(parent_dir, 'Imagenes-Clima'),
                    os.path.join(grandparent_dir, 'Imagenes-Clima')
                ]
                
                for ruta in otras_rutas:
                    if os.path.exists(ruta) and os.path.isdir(ruta):
                        directorio_encontrado = ruta
                        print(f"Directorio de imágenes encontrado en ruta alternativa: {ruta}")
                        break
            
            # Si encontramos el directorio, verificar que tenga al menos una imagen
            if directorio_encontrado:
                archivos = os.listdir(directorio_encontrado)
                imagenes = [f for f in archivos if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                if not imagenes:
                    print("ADVERTENCIA: Directorio de imágenes vacío o sin archivos de imagen")
                else:
                    print(f"Encontradas {len(imagenes)} imágenes en el directorio")
                    
                    # Intentar pre-cargar una imagen para verificar acceso
                    img_test_path = os.path.join(directorio_encontrado, imagenes[0])
                    try:
                        img = Image.open(img_test_path)
                        print(f"Prueba de acceso a imagen exitosa: {imagenes[0]}")
                    except Exception as e:
                        print(f"ERROR: No se pudo acceder a la imagen de prueba: {e}")
                        
        except Exception as e:
            print(f"Error verificando directorio de imágenes: {e}")
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
    def _on_configure(self, event=None):
        """Ajusta el contenido cuando la ventana cambia de tamaño"""
        try:
            # Evitar procesamiento si el evento no tiene dimensiones válidas
            if not event or not hasattr(event, 'width') or not hasattr(event, 'height'):
                return
                
            # Evitar actualización con tamaños muy pequeños
            if event.width < 100 or event.height < 100:
                return
                
            # Almacenar el último tamaño válido procesado para evitar actualizaciones innecesarias
            if hasattr(self, '_last_width') and hasattr(self, '_last_height'):
                # Si el cambio es menor a 10 píxeles, ignorar (reduce actualizaciones frecuentes)
                if (abs(self._last_width - event.width) < 10 and 
                    abs(self._last_height - event.height) < 10):
                    return
                    
            # Actualizar tamaño registrado
            self._last_width = event.width
            self._last_height = event.height
            
            # No hacer actualizaciones adicionales aquí para evitar problemas de renderizado
                    
        except Exception as e:
            # Manejo silencioso de errores para no interrumpir la interfaz
            print(f"Error en _on_configure: {e}")
    def fade_out_widgets(self):
        """Crea un efecto de desvanecimiento para widgets existentes"""
        try:
            for widget in self.main_frame.winfo_children():
                if widget.winfo_viewable():
                    # Bajar widget en la jerarquía de visualización (efecto de fade)
                    widget.lower()
                    # CORRECCIÓN: usar self en lugar de self.root
                    self.update_idletasks()
        except Exception as e:
            print(f"Advertencia en animación: {e}")
            pass  # Continuar si hay error en la animación
        
    def fade_out_widgets(self):
        """Crea un efecto de desvanecimiento para widgets existentes"""
        try:
            for widget in self.main_frame.winfo_children():
                if widget.winfo_viewable():
                    # Bajar widget en la jerarquía de visualización (efecto de fade)
                    widget.lower()
                    self.root.update_idletasks()
        except Exception as e:
            print(f"Advertencia en animación: {e}")
            pass  # Continuar si hay error en la animación
    def on_feedback_changed(self, event, fecha_periodo):
        """Maneja cuando el usuario cambia una categoría de clima"""
        try:
            fecha, periodo = fecha_periodo
            combo = event.widget
            nueva_categoria = combo.get()
            
            # Obtener predicción original
            widget_info = self.feedback_widgets[fecha_periodo]
            categoria_original = widget_info['categoria_original']
            
            print(f"Retroalimentación: {fecha.strftime('%Y-%m-%d')} - {periodo}")
            print(f"  Original: {categoria_original}")
            print(f"  Usuario: {nueva_categoria}")
            
            # Guardar esta retroalimentación
            self.guardar_retroalimentacion(fecha, periodo, categoria_original, nueva_categoria)
            
            # Efecto visual de confirmación mejorado
            inner_frame = widget_info.get('inner_frame', None)
            if inner_frame:
                bg_original = inner_frame.cget('bg')
                inner_frame.config(bg='#d0f0c0')  # Verde suave
                
                # Efecto de parpadeo suave
                def revert_bg():
                    inner_frame.config(bg='#e0ffe0')  # Verde más claro
                    self.after(150, lambda: inner_frame.config(bg=bg_original))
                    
                self.after(150, revert_bg)
            
            # Crear mensaje de progreso
            progreso_label = ttk.Label(self.main_frame, 
                                    text="Actualizando modelo con tu observación...",
                                    font=('Arial', 11, 'italic'),
                                    foreground='#0066cc')
            progreso_label.place(relx=0.5, rely=0.1, anchor='center')
            
            # Actualizar el modelo en tiempo real con mayor peso para observaciones humanas
            def actualizar_y_refrescar():
                try:
                    # Pasar un peso mucho mayor (30.0) para las observaciones humanas verificadas
                    self.actualizar_modelo_con_retroalimentacion(fecha, periodo, nueva_categoria, peso_observacion=30.0)
                    
                    # Refrescar vista después de actualizar el modelo para mostrar cambios visuales
                    self.after(100, lambda: self.eliminar_mensaje(progreso_label))
                    self.after(500, self.refresh_view)  # Refrescar vista después de un breve retardo
                    
                    # Mostrar mensaje de confirmación
                    self.after(600, lambda: messagebox.showinfo(
                        "Observación Registrada", 
                        f"Tu observación de '{nueva_categoria}' ha sido registrada y tendrá un peso importante en futuras predicciones."
                    ))
                except Exception as e:
                    print(f"Error en actualización: {e}")
                    self.after(100, lambda: self.eliminar_mensaje(progreso_label))
                    self.after(200, lambda: messagebox.showerror(
                        "Error", 
                        f"Ocurrió un error al actualizar el modelo: {str(e)}"
                    ))
            
            # Ejecutar en hilo separado
            threading.Thread(target=actualizar_y_refrescar, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar retroalimentación: {str(e)}")
        
    def eliminar_mensaje(self, widget):
        """Elimina suavemente un widget de mensaje"""
        if widget and widget.winfo_exists():
            widget.destroy()
    def simplificar_categoria(self, categoria):
        """Obtiene una versión simplificada de la categoría para mostrar en la visualización"""
        # Mapeamos las categorías a nombres más cortos
        if "Frío" in categoria:
            return "Frío"
        elif "Lluvia Fuerte" in categoria:
            return "Lluvia Fuerte"
        elif "Llovizna" in categoria:
            return "Llovizna"
        elif "Muy Nublado" in categoria:
            return "Nublado"
        elif "Parcialmente Nublado" in categoria:
            return "Parc. Nublado"
        elif "Normal" in categoria:
            return "Normal"
        elif "Templado" in categoria:
            return "Templado"
        elif "Cálido" in categoria:
            return "Cálido"
        else:
            # Tomar el primer componente de la categoría
            partes = categoria.split(' + ')
            if partes:
                return partes[0]
            return categoria
    def verificar_imagenes_clima(self, visualizador):
        """Verifica el estado de las imágenes del clima"""
        try:
            # Mensaje informativo
            mensaje = "Verificación de imágenes del clima:\n\n"
            
            # Mostrar directorio de trabajo actual
            directorio_actual = os.getcwd()
            mensaje += f"Directorio de trabajo actual: {directorio_actual}\n\n"
            
            # Probar diferentes rutas para encontrar la carpeta
            rutas_posibles = [
                'Imagenes-Clima',                                # Relativa al directorio actual
                os.path.join('..', 'Imagenes-Clima'),            # Directorio padre
                os.path.join(os.path.dirname(__file__), 'Imagenes-Clima'), # Directorio del script
                os.path.abspath('Imagenes-Clima')                # Ruta absoluta
            ]
            
            # Buscar la carpeta
            carpeta_encontrada = None
            mensaje += "Buscando carpeta en posibles ubicaciones:\n"
            for ruta in rutas_posibles:
                if os.path.exists(ruta):
                    carpeta_encontrada = ruta
                    mensaje += f"✅ ENCONTRADA en: {ruta}\n"
                else:
                    mensaje += f"❌ No encontrada en: {ruta}\n"
                    
            if not carpeta_encontrada:
                mensaje += "\n⚠️ La carpeta 'Imagenes-Clima' no fue encontrada."
                mensaje += "\n\nContenido del directorio actual:\n"
                try:
                    for item in os.listdir(directorio_actual):
                        mensaje += f"- {item}\n"
                except Exception as e:
                    mensaje += f"Error al listar archivos: {str(e)}\n"
            else:
                # Listar contenido
                mensaje += f"\nArchivos en la carpeta encontrada:\n"
                try:
                    archivos = os.listdir(carpeta_encontrada)
                    for archivo in archivos:
                        mensaje += f"- {archivo}\n"
                        
                    # Verificar archivos esperados (incluyendo nuevas imágenes nocturnas)
                    archivos_esperados = [
                        'Frio.png', 'Fuerte_Lluvia.png', 'Llovizna.png', 
                        'Nublado.png', 'Parcialmente_Soleado.png', 'Soleado.png',
                        'Noche_Despejada.png', 'Noche_Parcialmente_Nublado.png', 'Noche_Llovizna.png'
                    ]
                    
                    mensaje += "\nArchivos necesarios para la visualización:\n"
                    for esperado in archivos_esperados:
                        ruta = os.path.join(carpeta_encontrada, esperado)
                        if os.path.exists(ruta):
                            mensaje += f"✅ {esperado} - Encontrado\n"
                        else:
                            mensaje += f"⚠️ {esperado} - No encontrado\n"
                            
                    # Probar carga de imágenes nocturnas
                    mensaje += "\nPrueba de carga de imágenes nocturnas:\n"
                    categorias_prueba = ["Noche Despejada", "Noche Parcialmente Nublado", "Noche Llovizna"]
                    
                    # Crear fecha nocturna para prueba
                    fecha_nocturna = datetime.now().replace(hour=22, minute=0)
                    
                    for cat in categorias_prueba:
                        try:
                            img = visualizador.get_weather_icon(cat, fecha_nocturna)
                            if img is not None:
                                mensaje += f"✅ {cat} - Carga exitosa\n"
                            else:
                                mensaje += f"❌ {cat} - Fallo al cargar imagen\n"
                        except Exception as e:
                            mensaje += f"❌ {cat} - Error: {str(e)}\n"
                    
                except Exception as e:
                    mensaje += f"Error al listar archivos en carpeta: {str(e)}\n"
            
            # Mostrar mensaje
            messagebox.showinfo("Verificación de Imágenes", mensaje)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al verificar imágenes: {str(e)}")
            return False
    def on_closing(self):
        """Maneja el cierre de la ventana"""
        self.destroy()

def apply_custom_theme(root):
    # Colores de tema personalizados
    PRIMARY_COLOR = "#1976D2"      # Azul primario
    SECONDARY_COLOR = "#64B5F6"    # Azul secundario más claro
    ACCENT_COLOR = "#FF9800"       # Naranja para acentos
    BG_COLOR = "#F5F5F7"           # Fondo claro
    FRAME_BG = "#FFFFFF"           # Fondo de frames
    
    style = ttk.Style()
    
    # Configurar estilo general
    style.configure("TFrame", background=FRAME_BG)
    style.configure("TLabel", background=FRAME_BG, font=('Arial', 10))
    style.configure("TButton", font=('Arial', 10, 'bold'))
    
    # Botones con efectos al pasar el mouse
    style.map("TButton",
        background=[('active', SECONDARY_COLOR), ('pressed', PRIMARY_COLOR)],
        foreground=[('pressed', 'white'), ('active', 'white')])
    
    # Crear estilos específicos
    style.configure("Title.TLabel", font=('Arial', 12, 'bold'), foreground=PRIMARY_COLOR)
    style.configure("Section.TLabelframe", background=FRAME_BG)
    style.configure("Section.TLabelframe.Label", font=('Arial', 11, 'bold'), foreground=PRIMARY_COLOR)
    
    # Configurar color de fondo principal
    root.configure(background=BG_COLOR)
    
    # Devolver los colores para usarlos en otras partes
    return {
        "primary": PRIMARY_COLOR,
        "secondary": SECONDARY_COLOR,
        "accent": ACCENT_COLOR,
        "bg": BG_COLOR,
        "frame_bg": FRAME_BG
    }

def create_card_frame(parent, title):
    """Crea un frame con efecto de tarjeta elevada que usa todo el ancho disponible"""
    # Frame exterior para la sombra
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Configurar el frame para expandirse
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    
    # Frame principal con título
    card = ttk.LabelFrame(frame, text=title, style="Section.TLabelframe")
    card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    # Configurar el card para expandirse
    card.columnconfigure(0, weight=1)
    for i in range(10):  # Configurar varias filas para futura expansión
        card.rowconfigure(i, weight=1)
    
    return card

def create_temperature_display(parent, temperature=None, confidence=None, colors=None):
    """Crea un widget visualmente atractivo para mostrar la temperatura"""
    if colors is None:
        colors = {"primary": "#1976D2"}
        
    temp_frame = ttk.Frame(parent)
    temp_frame.pack(pady=10)
    
    # Canvas para dibujar el termómetro
    canvas = tk.Canvas(temp_frame, width=40, height=100, highlightthickness=0)
    canvas.pack(side=tk.LEFT, padx=10)
    
    # Dibujar termómetro
    # Base del termómetro
    canvas.create_oval(10, 80, 30, 100, fill="red", outline="")
    # Tubo
    canvas.create_rectangle(15, 20, 25, 90, fill="white", outline="")
    # Líquido
    if temperature is not None and temperature != "--°C":
        # Calcula altura basada en temperatura (rango de 5-30°C)
        if isinstance(temperature, str):
            temp_val = float(temperature.replace("°C", "").strip())
        else:
            temp_val = float(temperature)
        level = max(0, min(70, (temp_val - 5) * 3))
        canvas.create_rectangle(15, 90-level, 25, 90, fill="red", outline="")
    
    # Texto de temperatura
    temp_label = ttk.Label(temp_frame, 
                           text=temperature if temperature else "--°C", 
                           font=('Arial', 24, 'bold'),
                           foreground=colors["primary"])
    temp_label.pack(side=tk.LEFT, padx=10)
    
    # Confianza
    conf_text = f"Confianza: {confidence}" if confidence else "Confianza: --%"
    conf_frame = ttk.Frame(temp_frame)
    conf_frame.pack(side=tk.LEFT)
    
    conf_label = ttk.Label(conf_frame, text=conf_text)
    conf_label.pack(side=tk.TOP, anchor=tk.W)
    
    return temp_frame

def create_animated_progress_bar(parent):
    """Crea una barra de progreso con animación"""
    progress_frame = ttk.Frame(parent)
    progress_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Etiqueta de estado
    status_label = ttk.Label(progress_frame, text="Listo", anchor=tk.W)
    status_label.pack(side=tk.TOP, fill=tk.X)
    
    # Barra de progreso
    progress = ttk.Progressbar(progress_frame, mode='indeterminate')
    progress.pack(side=tk.TOP, fill=tk.X)
    
    def start_progress(message="Procesando..."):
        status_label.config(text=message)
        progress.start(10)
        progress_frame.update()
        
    def stop_progress(message="Completado"):
        progress.stop()
        status_label.config(text=message)
        progress_frame.update()
    
    # Exponer métodos para controlar la barra
    progress_frame.start = start_progress
    progress_frame.stop = stop_progress
    
    return progress_frame
class MicroClimaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción Meteorológica Udec Lestoma")
        self.root.geometry("850x950")
        self.root.minsize(650, 850)
        
        # Aplicar tema personalizado
        self.colors = apply_custom_theme(root)
        self.ui_scale = 1.0
        
        # Inicializar modelos y visualizadores
        self.predictor = PrediccionMeteo()
        self.visualizador = VisualizacionMicroclima()
        
        # Inicializar variables de ventanas
        self.ventana_viz = None
        self.ventana_pronostico = None
        self.ventana_progreso = None
        
        # Definir pred_frame
        self.pred_frame = None
        
        # CAMBIO IMPORTANTE: Configurar el root para expandirse correctamente
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Crear un frame principal contenedor
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Crear un canvas con scrollbar para contener todo
        self.canvas = tk.Canvas(self.main_container, bg=self.colors["bg"])
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="TFrame")
        
        # Configurar el scrollable_frame para expandirse al tamaño del canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        # Añadir el frame al canvas - MODIFICADO para usar todo el ancho disponible
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # CAMBIO: Usar grid en lugar de pack para mejor control de la expansión
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configurar el redimensionamiento del canvas_window cuando cambia el tamaño del canvas
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
        # Variables de estado
        self.dataset = None
        self.ultimo_modelo = None
        self.entrenamiento_activo = False
        
        # Crear interfaz mejorada
        self.create_header()
        self.create_main_interface()
        
        # Configurar estilo
        self.configure_styles()
        
        # Configurar evento de scroll con rueda del ratón
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<Button-4>", self._on_mousewheel)
        self.root.bind("<Button-5>", self._on_mousewheel)
        self.root_width = 850
        self.root_height = 950
    def _on_canvas_resize(self, event):
        """Ajusta el ancho del contenido del canvas cuando éste se redimensiona"""
        # Ajustar el ancho del frame interno al ancho del canvas
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        # Forzar actualización
        self.root.update_idletasks()
    def create_header(self):
        """Crea un encabezado atractivo para la aplicación"""
        header = ttk.Frame(self.scrollable_frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        # Logo (simulado con Canvas)
        logo_canvas = tk.Canvas(header, width=60, height=60, highlightthickness=0, bg=self.colors["bg"])
        logo_canvas.pack(side=tk.LEFT, padx=10)
        
        # Dibujar logo simple
        logo_canvas.create_oval(5, 5, 55, 55, fill=self.colors["primary"], outline="")
        logo_canvas.create_text(30, 30, text="MT", fill="white", font=('Arial', 18, 'bold'))
        
        # Título y subtítulo
        title_frame = ttk.Frame(header)
        title_frame.pack(side=tk.LEFT, padx=10)
        
        title = ttk.Label(title_frame, 
                        text="Sistema de Predicción Meteorológica Udec Lestoma", 
                        font=('Arial', 16, 'bold'),
                        foreground=self.colors["primary"])
        title.pack(anchor=tk.W)
        
        subtitle = ttk.Label(title_frame, 
                            text="Universidad de Cundinarma Facatativá, Colombia", 
                            font=('Arial', 12))
        subtitle.pack(anchor=tk.W)
        
        # Fecha actual
        date_label = ttk.Label(header, 
                            text=datetime.now().strftime("%d de %B de %Y"),
                            font=('Arial', 10, 'italic'))
        date_label.pack(side=tk.RIGHT, padx=10)
        config_button = ttk.Button(
            header, 
            text="⚙️",
            width=3,
            command=self.open_config_window
        )
        config_button.pack(side=tk.RIGHT, padx=5)
    def open_config_window(self):
        """Abre una ventana de configuración para ajustar la resolución"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Configuración de Visualización")
        config_window.geometry("400x350")  # Aumentado para asegurar que todos los elementos sean visibles
        config_window.resizable(False, False)
        
        # Centrar ventana
        config_window.update_idletasks()
        width = config_window.winfo_width()
        height = config_window.winfo_height()
        x = (config_window.winfo_screenwidth() // 2) - (width // 2)
        y = (config_window.winfo_screenheight() // 2) - (height // 2)
        config_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # Frame principal
        main_frame = ttk.Frame(config_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(
            main_frame, 
            text="Ajustes de Visualización", 
            font=('Arial', 14, 'bold'),
            foreground=self.colors["primary"]
        ).pack(pady=10)
        
        # Resolución principal
        resolution_frame = ttk.LabelFrame(main_frame, text="Resolución de Ventana Principal", padding=10)
        resolution_frame.pack(fill=tk.X, pady=10)
        
        # Obtener tamaño actual
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        
        # Opciones predefinidas de resolución
        ttk.Label(resolution_frame, text="Tamaño de ventana:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Variable para almacenar la selección
        self.resolution_var = tk.StringVar(value=f"{current_width}x{current_height}")
        
        # Opciones de resolución basadas en el tamaño de la pantalla
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        resolutions = [
            f"{screen_width}x{screen_height}",  # Pantalla completa
            f"{int(screen_width*0.8)}x{int(screen_height*0.8)}",  # 80% de la pantalla
            f"{int(screen_width*0.7)}x{int(screen_height*0.7)}",  # 70% de la pantalla
            f"{int(screen_width*0.6)}x{int(screen_height*0.6)}",  # 60% de la pantalla
            f"{int(screen_width*0.5)}x{int(screen_height*0.5)}",  # 50% de la pantalla
            "850x950",  # Resolución original
            "800x800",  # Más pequeño
            "700x800",  # Aún más pequeño
            "Personalizado"
        ]
        
        resolution_combo = ttk.Combobox(
            resolution_frame, 
            textvariable=self.resolution_var,
            values=resolutions,
            width=20,
            state="readonly"
        )
        resolution_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        resolution_combo.bind("<<ComboboxSelected>>", self.on_resolution_selected)
        
        # Frame para resolución personalizada
        custom_frame = ttk.Frame(resolution_frame)
        custom_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(custom_frame, text="Ancho:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.custom_width_var = tk.StringVar(value=str(current_width))
        width_entry = ttk.Entry(custom_frame, textvariable=self.custom_width_var, width=8)
        width_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(custom_frame, text="Alto:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.custom_height_var = tk.StringVar(value=str(current_height))
        height_entry = ttk.Entry(custom_frame, textvariable=self.custom_height_var, width=8)
        height_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Escala de interfaces secundarias
        scale_frame = ttk.LabelFrame(main_frame, text="Escala de Interfaces Secundarias", padding=10)
        scale_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(scale_frame, text="Tamaño relativo:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(
            scale_frame, 
            from_=0.7, 
            to=1.3, 
            variable=self.scale_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        scale_slider.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Etiqueta para mostrar valor actual
        self.scale_label = ttk.Label(scale_frame, text="100%")
        self.scale_label.grid(row=0, column=2, padx=5)
        
        # Actualizador de valor
        def update_scale_label(*args):
            self.scale_label.config(text=f"{int(self.scale_var.get() * 100)}%")
        
        self.scale_var.trace_add("write", update_scale_label)
        
        # Botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        ttk.Button(
            button_frame, 
            text="Aplicar", 
            command=self.apply_config
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Cancelar", 
            command=config_window.destroy
        ).pack(side=tk.RIGHT, padx=5)

    def on_resolution_selected(self, event):
        """Maneja el cambio de resolución seleccionada"""
        selected = self.resolution_var.get()
        if selected == "Personalizado":
            # No hacer nada, dejar que el usuario introduzca valores personalizados
            pass
        else:
            # Actualizar campos personalizados
            width, height = map(int, selected.split("x"))
            self.custom_width_var.set(str(width))
            self.custom_height_var.set(str(height))

    def apply_config(self):
        """Aplica la configuración seleccionada"""
        try:
            # Obtener nuevas dimensiones
            if self.resolution_var.get() == "Personalizado":
                width = int(self.custom_width_var.get())
                height = int(self.custom_height_var.get())
            else:
                width, height = map(int, self.resolution_var.get().split("x"))
            
            # Asegurarse de que los valores son razonables
            width = max(500, min(width, self.root.winfo_screenwidth()))
            height = max(600, min(height, self.root.winfo_screenheight()))
            
            # IMPORTANTE: Aplicar con update para forzar el cambio inmediato
            self.root.geometry(f"{width}x{height}")
            self.root.update_idletasks()
            
            # Guardar escala para ventanas secundarias
            self.ui_scale = self.scale_var.get()
            
            # Cerrar todas las ventanas secundarias para que se reabran con la nueva escala
            if hasattr(self, 'ventana_viz') and self.ventana_viz and self.ventana_viz.winfo_exists():
                self.ventana_viz.destroy()
                
            if hasattr(self, 'ventana_pronostico') and self.ventana_pronostico and self.ventana_pronostico.winfo_exists():
                self.ventana_pronostico.destroy()
            
            # AÑADIDO: Centrar ventana principal después del cambio
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            
            # AÑADIDO: Guardar configuración en atributos de la clase
            self.root_width = width
            self.root_height = height
            
            # Mensaje de confirmación
            messagebox.showinfo(
                "Configuración Aplicada", 
                f"Se ha aplicado la nueva configuración:\nResolución: {width}x{height}\nEscala: {int(self.ui_scale * 100)}%"
            )
            
        except ValueError as e:
            messagebox.showerror("Error", f"Los valores de resolución deben ser números enteros.\nError: {str(e)}")
    def _on_mousewheel(self, event):
        """Maneja el evento de la rueda del ratón para el scroll"""
        # Para Windows/MacOS
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
    def create_data_update_area(self, parent):
        """Crea la sección de actualización de datos con Maquina del Tiempo"""
        update_frame = ttk.LabelFrame(parent, text="Actualización con Datos de Estación", padding="5")
        update_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        
        # Botón de actualización manual
        ttk.Button(update_frame, 
                text="Actualizar con Nuevos Datos",
                command=self.update_with_station_data, 
                width=30).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Estado de última actualización
        self.last_update_label = ttk.Label(update_frame, 
                                        text="Última actualización: Nunca", 
                                        style='Info.TLabel')
        self.last_update_label.pack(side=tk.LEFT, padx=20, pady=5)
        
    def update_with_station_data(self):
        """Actualiza el modelo con los datos más recientes de la estación"""
        try:
            # Mostrar mensaje explicativo antes de iniciar el proceso
            messagebox.showinfo(
                "Proceso de Actualización",
                "El proceso de actualización requiere dos archivos:\n\n"
                "1. PRIMERO se te pedirá seleccionar el DATASET HISTÓRICO COMPLETO\n"
                "   (Debe contener todos los datos desde 2018 hasta la actualidad)\n\n"
                "2. DESPUÉS se te pedirá seleccionar el archivo con los NUEVOS DATOS\n"
                "   (Solo los datos recientes que deseas incorporar)"
            )

            # Solicitar archivo histórico con mensaje claro
            ruta_historico = self.dataset_path if hasattr(self, 'dataset_path') else None
            if not ruta_historico:
                ruta_historico = filedialog.askopenfilename(
                    title="PASO 1: Seleccionar DATASET HISTÓRICO COMPLETO (2018-actualidad)",
                    filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
                )
                if not ruta_historico:
                    return
            
            # Solicitar archivo con nuevos datos, con mensaje claro
            ruta_estacion = filedialog.askopenfilename(
                title="PASO 2: Seleccionar ARCHIVO CON NUEVOS DATOS (solo los datos recientes)",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if not ruta_estacion:
                return
            
            # Confirmar las selecciones
            confirmacion = f"Has seleccionado:\n\n" \
                        f"• Dataset histórico: {os.path.basename(ruta_historico)}\n" \
                        f"• Nuevos datos: {os.path.basename(ruta_estacion)}\n\n" \
                        f"¿Son correctos estos archivos?"
            
            if not messagebox.askyesno("Confirmar selección", confirmacion):
                messagebox.showinfo("Proceso cancelado", "Puedes volver a intentarlo seleccionando los archivos correctos.")
                return
            
            # Mostrar ventana de progreso
            self.ventana_progreso = VentanaProgreso(self.root, "Actualizando Modelo")
            self.ventana_progreso.update_progress(0, "Iniciando actualización...")
            
            def update_task():
                try:
                    # 1. Preprocesar datos de estación para marcarlos como verificados
                    self.ventana_progreso.update_progress(10, "Preparando datos de estación...")
                    
                    # Cargar datos de estación
                    df_estacion = pd.read_csv(ruta_estacion)
                    
                    # Verificar el formato de la fecha
                    if 'fecha_hora' in df_estacion.columns:
                        # Convertir columna fecha_hora a formato estándar
                        df_estacion['fecha'] = pd.to_datetime(df_estacion['fecha_hora'])
                    elif 'fecha' in df_estacion.columns:
                        # Ya tiene la columna correcta
                        df_estacion['fecha'] = pd.to_datetime(df_estacion['fecha'])
                    else:
                        raise ValueError("No se encontró una columna de fecha válida en los datos de la estación")
                    
                    # Mapear columnas a formato estándar
                    df_procesado = pd.DataFrame()
                    df_procesado['fecha'] = df_estacion['fecha']
                    
                    # Mapear temperatura según columnas disponibles
                    if 'temp_dht_cal' in df_estacion.columns:
                        df_procesado['temperatura_C'] = df_estacion['temp_dht_cal']  # Usar temperatura calibrada
                    elif 'temperatura_C' in df_estacion.columns:
                        df_procesado['temperatura_C'] = df_estacion['temperatura_C']
                    else:
                        df_procesado['temperatura_C'] = df_estacion['temp_dht_raw']  # Alternativa
                    
                    # Mapear humedad según columnas disponibles
                    if 'hum_dht_cal' in df_estacion.columns:
                        df_procesado['humedad_relativa'] = df_estacion['hum_dht_cal']  # Usar humedad calibrada
                    elif 'humedad_relativa' in df_estacion.columns:
                        df_procesado['humedad_relativa'] = df_estacion['humedad_relativa']
                    else:
                        df_procesado['humedad_relativa'] = df_estacion['hum_dht_raw']  # Alternativa
                    
                    # Mapear precipitación
                    if 'lluvia_actual_mm' in df_estacion.columns:
                        df_procesado['precipitacion_mm'] = df_estacion['lluvia_actual_mm']
                    elif 'precipitacion_mm' in df_estacion.columns:
                        df_procesado['precipitacion_mm'] = df_estacion['precipitacion_mm']
                    else:
                        df_procesado['precipitacion_mm'] = df_estacion['lluvia_mm']  # Alternativa
                    
                    # Mapear nubosidad
                    if 'cobertura_nubes_octas' in df_estacion.columns:
                        df_procesado['cobertura_nubes_octas'] = df_estacion['cobertura_nubes_octas']
                    else:
                        # Calcular nubosidad si no está disponible
                        print("Columna de nubosidad no encontrada, utilizando valor por defecto")
                        df_procesado['cobertura_nubes_octas'] = 4.0  # Valor por defecto
                    
                    # Mapear velocidad del viento
                    if 'vel_viento_kmh' in df_estacion.columns:
                        df_procesado['velocidad_viento_kmh'] = df_estacion['vel_viento_kmh']
                    elif 'velocidad_viento_kmh' in df_estacion.columns:
                        df_procesado['velocidad_viento_kmh'] = df_estacion['velocidad_viento_kmh']
                    else:
                        df_procesado['velocidad_viento_kmh'] = 0.0  # Valor por defecto
                    
                    # Mapear radiación solar
                    if 'radiacion_solar_wm2' in df_estacion.columns:
                        # Convertir de W/m² a J/m²
                        df_procesado['radiacion_solar_J_m2'] = df_estacion['radiacion_solar_wm2'] * 3600
                    elif 'radiacion_solar_J_m2' in df_estacion.columns:
                        df_procesado['radiacion_solar_J_m2'] = df_estacion['radiacion_solar_J_m2']
                    else:
                        # Estimar a partir de luminosidad si está disponible
                        if 'luminosidad_lux' in df_estacion.columns:
                            # Conversión aproximada
                            df_procesado['radiacion_solar_J_m2'] = df_estacion['luminosidad_lux'] * 0.0079 * 3600
                        else:
                            # Valor por defecto basado en la hora del día
                            df_procesado['radiacion_solar_J_m2'] = df_procesado['fecha'].apply(
                                lambda x: 800*3600 if 6 <= x.hour <= 18 else 0
                            )
                    
                    # Calcular categoría climática usando el método del predictor
                    self.ventana_progreso.update_progress(25, "Calculando categorías climáticas...")
                    
                    # Crear lista para almacenar las categorías
                    categorias = []
                    
                    # Calcular categoría para cada fila
                    for idx, row in df_procesado.iterrows():
                        # Usar el método de categorización del predictor
                        categoria = self.predictor.categorizar_clima(row)
                        categorias.append(categoria)
                    
                    # Añadir categorías calculadas al dataframe
                    df_procesado['categoria_clima'] = categorias
                    
                    # Añadir columna que indica datos verificados
                    df_procesado['verificado'] = True
                    
                    # Establecer fechas como índice
                    df_procesado.set_index('fecha', inplace=True)
                    
                    # Guardar datos procesados temporalmente
                    ruta_temp = "datos_estacion_procesados.csv"
                    df_procesado.to_csv(ruta_temp)
                    
                    self.ventana_progreso.update_progress(30, "Datos de estación procesados...")
                    
                    # 2. Integrar con datos históricos
                    self.ventana_progreso.update_progress(40, "Integrando datasets...")
                    dataset_completo = integrar_datasets(
                        ruta_historico,
                        ruta_temp,
                        "dataset_completo_actualizado.csv"
                    )
                    
                    # 3. Actualizar modelo con mayor peso a datos verificados
                    self.ventana_progreso.update_progress(60, "Actualizando modelo...")
                    
                    # Función para asignar pesos según si los datos están verificados
                    def sample_weights(df):
                        if 'verificado' in df.columns:
                            return np.where(df['verificado'] == True, 3.0, 1.0)
                        else:
                            return np.ones(len(df))
                    
                    # Actualizar modelo con los datos integrados y pesos
                    history = self.predictor.actualizar_modelo_con_nuevos_datos(
                        "dataset_completo_actualizado.csv",
                        guardar=True,
                        sample_weights=sample_weights
                    )
                    
                    # 4. Actualizar el dataset principal
                    self.dataset = dataset_completo
                    self.dataset_path = "dataset_completo_actualizado.csv"
                    
                    # 5. Actualizar interfaz
                    self.ventana_progreso.update_progress(90, "Finalizando...")
                    self.root.after(0, lambda history=history: self.after_model_update(history))
                    
                    # 6. Mostrar mensaje de éxito con información sobre los datos
                    def mostrar_resumen():
                        messagebox.showinfo(
                            "Actualización Completada",
                            f"La actualización se ha completado exitosamente:\n\n"
                            f"• Dataset completo actualizado: {len(dataset_completo)} registros\n"
                            f"• Rango de fechas: {dataset_completo.index.min().strftime('%Y-%m-%d')} hasta "
                            f"{dataset_completo.index.max().strftime('%Y-%m-%d')}\n\n"
                            f"El modelo ha sido actualizado para incorporar los nuevos datos verificados."
                        )
                    
                    self.root.after(200, mostrar_resumen)
                    
                except Exception as err:
                    self.root.after(0, lambda err=err: self.show_error(
                        "Error en actualización", 
                        str(err)
                    ))
                finally:
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
            
            # Ejecutar tarea en segundo plano
            threading.Thread(target=update_task, daemon=True).start()
            
        except Exception as err:
            self.show_error("Error", f"Error al iniciar actualización: {str(err)}")
            
    def after_model_update(self, history):
        """Acciones después de actualizar el modelo"""
        # Actualizar etiqueta de última actualización
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.last_update_label.config(text=f"Última actualización: {now}")
        
        # Actualizar log de entrenamiento
        self.train_log.insert(tk.END, f"\nActualización completada: {now}\n")
        
        # Mostrar mensaje de éxito
        messagebox.showinfo("Actualización Completada", 
                            "El modelo ha sido actualizado con los datos más recientes de la estación meteorológica.")
    def configure_styles(self):
        """Configura los estilos de la interfaz"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Info.TLabel', font=('Arial', 9))
        
    def center_window(self):
        """Centra la ventana en la pantalla usando dimensiones guardadas"""
        width = getattr(self, 'root_width', self.root.winfo_width())
        height = getattr(self, 'root_height', self.root.winfo_height())
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def create_main_interface(self):
        """Crea la interfaz principal con tarjetas elevadas que se expanden correctamente"""
        # Configurar scrollable_frame para que se expanda bien
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # Barra de progreso animada
        self.progress_bar = create_animated_progress_bar(self.scrollable_frame)
        
        # Sección de selección de datos
        data_card = create_card_frame(self.scrollable_frame, "Selección de Datos")
        self.create_file_selection(data_card)
        
        # Sección de información del dataset
        info_card = create_card_frame(self.scrollable_frame, "Información del Dataset")
        self.create_data_display(info_card)
        
        # Sección de entrenamiento del modelo
        train_card = create_card_frame(self.scrollable_frame, "Entrenamiento del Modelo")
        self.create_training_area(train_card)
        
        # Sección de predicciones
        pred_card = create_card_frame(self.scrollable_frame, "Predicciones")
        self.create_prediction_area(pred_card)
        
        # Sección de visualizaciones
        viz_card = create_card_frame(self.scrollable_frame, "Visualizaciones")
        self.create_visualization_area(viz_card)
        
        # Sección de actualización de datos
        update_card = create_card_frame(self.scrollable_frame, "Actualización con Datos de Estación")
        self.create_data_update_area(update_card)
        
        # Sección de estación meteorológica
        station_card = create_card_frame(self.scrollable_frame, "Estación Meteorológica")
        self.create_station_tab(station_card)
    def create_file_selection(self, parent):
        """Crea la sección de selección de archivos con estilo mejorado que usa todo el ancho"""
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configurar expansión
        file_frame.columnconfigure(0, weight=1)
        
        # Crear marco para el botón y etiqueta
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, expand=True)
        
        # Configurar expansión del input_frame
        input_frame.columnconfigure(1, weight=1)
        
        # Botón de selección con ícono
        select_frame = ttk.Frame(input_frame)
        select_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Simular ícono con Canvas
        icon_canvas = tk.Canvas(select_frame, width=20, height=20, 
                            bg=self.colors["bg"], highlightthickness=0)
        icon_canvas.pack(side=tk.LEFT, padx=(0, 5))
        
        # Dibujar ícono de archivo
        icon_canvas.create_rectangle(5, 3, 15, 17, fill=self.colors["secondary"], outline="")
        icon_canvas.create_line(10, 3, 10, 17, fill="white")
        
        # Botón de selección
        ttk.Button(select_frame, 
                text="Seleccionar Dataset",
                command=self.load_dataset, 
                width=20).pack(side=tk.LEFT)
        
        # Etiqueta de archivo con fondo visual - MODIFICADO para usar todo el ancho
        file_label_frame = ttk.Frame(input_frame, padding=5)
        file_label_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Configurar para expandirse
        file_label_frame.columnconfigure(0, weight=1)
        
        self.file_label = ttk.Label(
            file_label_frame, 
            text="Ningún archivo seleccionado", 
            wraplength=400, 
            padding=5,
            background="#F0F0F0",
            relief="groove"
        )
        self.file_label.pack(fill=tk.X, expand=True)
        
    def create_data_display(self, parent):
        """Crea la sección de visualización de datos que usa todo el ancho"""
        data_frame = ttk.LabelFrame(parent, text="Información del Dataset", padding="5")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar grid para expandirse
        data_frame.grid_columnconfigure(0, weight=1)
        data_frame.grid_rowconfigure(0, weight=1)
        
        # Área de texto con scroll
        self.data_text = tk.Text(data_frame, 
                                height=8, 
                                width=70, 
                                font=('Consolas', 9))
        scroll = ttk.Scrollbar(data_frame, 
                            orient="vertical", 
                            command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=scroll.set)
        
        # Usar grid para mejor expansión
        self.data_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scroll.grid(row=0, column=1, sticky="ns")
        
    def create_training_area(self, parent):
        """Crea la sección de entrenamiento con parámetros ajustables del modelo"""
        train_frame = ttk.LabelFrame(parent, text="Entrenamiento del Modelo", padding="5")
        train_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Frame para botones
        button_frame = ttk.Frame(train_frame)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Botones de entrenamiento
        ttk.Button(button_frame, 
                text="Cargar Modelo",
                command=self.cargar_modelo, 
                width=20).grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, 
                text="Entrenar Modelo",
                command=self.train_model, 
                width=20).grid(row=0, column=1, padx=20)
        
        # Frame para parámetros básicos
        param_frame = ttk.Frame(train_frame)
        param_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Configuración de épocas
        ttk.Label(param_frame, text="Épocas:").grid(row=0, column=0, padx=5)
        self.epochs_var = tk.StringVar(value="200")
        ttk.Entry(param_frame, 
                textvariable=self.epochs_var, 
                width=8).grid(row=0, column=1, padx=5)
        
        # Configuración de batch size
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, padx=5)
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(param_frame, 
                textvariable=self.batch_size_var, 
                width=8).grid(row=0, column=3, padx=5)
        
        # Frame para parámetros avanzados
        advanced_frame = ttk.LabelFrame(train_frame, text="Parámetros Avanzados", padding="5")
        advanced_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Primera fila de parámetros avanzados
        adv_row1 = ttk.Frame(advanced_frame)
        adv_row1.pack(fill=tk.X, pady=3)
        
        # Tasa de aprendizaje
        ttk.Label(adv_row1, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(adv_row1, 
                textvariable=self.learning_rate_var, 
                width=8).pack(side=tk.LEFT, padx=5)
        
        # Unidades LSTM
        ttk.Label(adv_row1, text="Unidades LSTM:").pack(side=tk.LEFT, padx=5)
        self.lstm_units_var = tk.StringVar(value="96")
        ttk.Entry(adv_row1, 
                textvariable=self.lstm_units_var, 
                width=8).pack(side=tk.LEFT, padx=5)
        
        # Segunda fila de parámetros avanzados
        adv_row2 = ttk.Frame(advanced_frame)
        adv_row2.pack(fill=tk.X, pady=3)
        
        # Tasa de dropout
        ttk.Label(adv_row2, text="Dropout:").pack(side=tk.LEFT, padx=5)
        self.dropout_var = tk.StringVar(value="0.3")
        ttk.Entry(adv_row2, 
                textvariable=self.dropout_var, 
                width=8).pack(side=tk.LEFT, padx=5)
        
        # Regularización L2
        ttk.Label(adv_row2, text="Reg. L2:").pack(side=tk.LEFT, padx=5)
        self.l2_reg_var = tk.StringVar(value="0.01")
        ttk.Entry(adv_row2, 
                textvariable=self.l2_reg_var, 
                width=8).pack(side=tk.LEFT, padx=5)
        
        # Casilla de verificación para uso de ensemble
        self.use_ensemble_var = tk.BooleanVar(value=False)
        ensemble_check = ttk.Checkbutton(advanced_frame, 
                                        text="Usar Ensemble de Modelos", 
                                        variable=self.use_ensemble_var)
        ensemble_check.pack(fill=tk.X, pady=3)
        
        # Log de entrenamiento
        log_frame = ttk.Frame(train_frame)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        
        self.train_log = tk.Text(log_frame, 
                                height=5, 
                                width=70, 
                                font=('Consolas', 9))
        scroll = ttk.Scrollbar(log_frame, 
                            orient="vertical", 
                            command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=scroll.set)
        
        self.train_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
    def create_prediction_area(self, parent):
        """Crea la sección de predicciones"""
        pred_frame = ttk.Frame(parent)
        pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel superior para mostrar clima actual
        self.current_weather_frame = ttk.Frame(pred_frame)
        self.current_weather_frame.pack(fill=tk.X, pady=5)
        
        # Temperatura actual con visualización tradicional
        # (mantener para compatibilidad)
        self.current_temp_label = ttk.Label(self.current_weather_frame, 
                                        text="--°C", 
                                        font=('Arial', 24, 'bold'))
        self.current_temp_label.pack(side=tk.LEFT, padx=10)
        
        # Información actual
        current_info_frame = ttk.Frame(self.current_weather_frame)
        current_info_frame.pack(side=tk.LEFT, padx=10)
        
        self.current_condition_label = ttk.Label(current_info_frame, 
                                            text="--", 
                                            font=('Arial', 12))
        self.current_condition_label.pack(anchor=tk.W)
        
        self.current_confidence_label = ttk.Label(current_info_frame, 
                                                text="Confianza: --%",
                                                font=('Arial', 10))
        self.current_confidence_label.pack(anchor=tk.W)
        
        # Controles de predicción
        control_frame = ttk.Frame(pred_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón de predicción
        self.predict_button = ttk.Button(control_frame, 
                                    text="Generar Predicciones",
                                    command=self.generate_predictions, 
                                    width=20)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Botón de exportar predicciones
        self.export_button = ttk.Button(control_frame,
                                    text="Exportar Predicciones",
                                    command=self.export_predictions,
                                    width=20)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Frame para contener el área de texto y la barra de desplazamiento
        text_frame = ttk.Frame(pred_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Área de texto para predicciones
        self.pred_text = tk.Text(text_frame, 
                            height=8, 
                            width=70, 
                            font=('Consolas', 9))
        scroll = ttk.Scrollbar(text_frame, 
                            orient="vertical", 
                            command=self.pred_text.yview)
        self.pred_text.configure(yscrollcommand=scroll.set)
        
        self.pred_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
    def mostrar_visualizaciones(self, tipo="temperatura"):
        """Muestra las visualizaciones según el tipo seleccionado"""
        try:
            # Mostrar ventana de progreso
            self.ventana_progreso = VentanaProgreso(self.root, "Generando Visualización")
            self.ventana_progreso.update_progress(10, "Preparando datos...")
            
            # Ejecución en segundo plano
            def generate_visualization():
                try:
                    # Obtener predicciones primero para evitar errores posteriores
                    predicciones = None
                    if hasattr(self, 'predictor') and self.predictor.model is not None and hasattr(self, 'dataset'):
                        try:
                            self.ventana_progreso.update_progress(30, "Generando predicciones...")
                            predicciones = self.predictor.predecir_proximo_periodo(self.dataset)
                        except Exception as e:
                            self.root.after(0, lambda: self.show_error("Error", f"Error al generar predicciones: {str(e)}"))
                            self.ventana_progreso.safe_destroy()
                            return
                    else:
                        self.root.after(0, lambda: self.show_error("Advertencia", "No hay modelo cargado o dataset disponible para generar predicciones."))
                        self.ventana_progreso.safe_destroy()
                        return
                    
                    self.ventana_progreso.update_progress(50, "Procesando visualización...")
                    
                    # Gestionar visualizaciones según el tipo
                    if tipo == "temperatura":
                        # Visualización de temperatura
                        # Cerrar ventana anterior si existe
                        if hasattr(self, 'ventana_viz'):
                            try:
                                if isinstance(self.ventana_viz, tk.Toplevel) and self.ventana_viz.winfo_exists():
                                    self.ventana_viz.destroy()
                            except Exception:
                                pass
                                
                        self.ventana_progreso.update_progress(70, "Creando gráfica de temperatura...")
                        
                        # Crear nueva ventana
                        self.ventana_viz = VentanaVisualizacion(self.root)
                        self.ventana_viz.title("Pronóstico de Temperatura y Confianza")
                        
                        # Aplicar escala
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()
                        width = int(min(screen_width * 0.8, 1000) * self.ui_scale)
                        height = int(min(screen_height * 0.8, 600) * self.ui_scale)
                        self.ventana_viz.geometry(f"{width}x{height}")
                        
                        # Actualizar visualización con predicciones
                        if predicciones:
                            try:
                                # Crear gráfica de temperatura usando el visualizador
                                fig = self.visualizador.crear_grafica_temperatura(predicciones)
                                
                                # Limpiar frames de la ventana
                                for widget in self.ventana_viz.main_frame.winfo_children():
                                    widget.destroy()
                                    
                                # Crear canvas con la figura
                                canvas = FigureCanvasTkAgg(fig, self.ventana_viz.main_frame)
                                canvas.draw()
                                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                                
                                # Crear barra de navegación
                                if hasattr(self.ventana_viz, 'pred_frame') and self.ventana_viz.pred_frame:
                                    for widget in self.ventana_viz.pred_frame.winfo_children():
                                        widget.destroy()
                                else:
                                    self.ventana_viz.pred_frame = ttk.Frame(self.ventana_viz.main_frame)
                                    self.ventana_viz.pred_frame.pack(fill=tk.X)
                                    
                                toolbar = NavigationToolbar2Tk(canvas, self.ventana_viz.pred_frame)
                                toolbar.update()
                                
                            except Exception as e:
                                self.root.after(0, lambda: self.show_error("Error", f"Error al actualizar visualización de temperatura: {str(e)}"))
                                self.ventana_progreso.safe_destroy()
                                return
                        
                        # Traer ventana al frente
                        self.ventana_viz.lift()
                        
                    elif tipo == "detallado":
                        # Visualización detallada
                        # Cerrar ventana anterior si existe
                        if hasattr(self, 'ventana_pronostico'):
                            try:
                                if isinstance(self.ventana_pronostico, tk.Toplevel) and self.ventana_pronostico.winfo_exists():
                                    self.ventana_pronostico.destroy()
                            except Exception:
                                pass
                        
                        self.ventana_progreso.update_progress(70, "Creando pronóstico detallado...")
                        
                        # Crear nueva ventana
                        self.ventana_pronostico = VentanaPronosticoDetallado(self.root)
                        
                        # Aplicar escala
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()
                        width = int(min(screen_width * 0.8, 800) * self.ui_scale)
                        height = int(min(screen_height * 0.8, 600) * self.ui_scale)
                        self.ventana_pronostico.geometry(f"{width}x{height}")
                        
                        # IMPORTANTE: Asegurar que el visualizador tenga acceso al predictor
                        self.visualizador.predictor = self.predictor
                        
                        # Actualizar visualización con predicciones
                        if predicciones:
                            try:
                                # Asegurar que la ventana tenga el atributo pred_frame
                                if not hasattr(self.ventana_pronostico, 'pred_frame'):
                                    self.ventana_pronostico.pred_frame = ttk.Frame(self.ventana_pronostico)
                                    self.ventana_pronostico.pred_frame.pack(fill=tk.X, padx=5, pady=2)
                                
                                # Actualizar la gráfica
                                self.ventana_pronostico.actualizar_grafica(predicciones, self.visualizador)
                                
                                # Verificar la visibilidad después de un breve retraso
                                self.ventana_pronostico.after(500, self.ventana_pronostico.verificar_visibilidad)
                                
                            except Exception as e:
                                self.root.after(0, lambda: self.show_error("Error", f"Error al actualizar pronóstico detallado: {str(e)}"))
                                self.ventana_progreso.safe_destroy()
                                return
                        
                        # Traer ventana al frente
                        self.ventana_pronostico.lift()
                        
                    else:
                        print(f"Tipo de visualización no reconocido: {tipo}")
                    
                    # Completar proceso y cerrar ventana de progreso
                    self.ventana_progreso.update_progress(100, "¡Visualización completada!")
                    self.root.after(500, self.ventana_progreso.safe_destroy)
                    
                except Exception as e:
                    self.root.after(0, lambda: self.show_error("Error", f"Error al mostrar visualización: {str(e)}"))
                    if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
            
            # Ejecutar en segundo plano para mantener la UI responsiva
            threading.Thread(target=generate_visualization, daemon=True).start()
            
        except Exception as e:
            self.show_error("Error", f"Error al iniciar la visualización: {str(e)}")
            if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                self.ventana_progreso.safe_destroy()
    def create_visualization_area(self, parent):
        """Crea la sección de visualizaciones"""
        viz_frame = ttk.LabelFrame(parent, text="Visualizaciones", padding="5")
        viz_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón de visualización de temperatura
        ttk.Button(button_frame, 
                  text="Ver Temperatura",
                  command=lambda: self.mostrar_visualizaciones("temperatura"), 
                  width=20).pack(side=tk.LEFT, padx=5)
        
        # Botón de visualización detallada
        ttk.Button(button_frame, 
                  text="Ver Pronóstico Detallado",
                  command=lambda: self.mostrar_visualizaciones("detallado"), 
                  width=25).pack(side=tk.LEFT, padx=5)
    def cargar_modelo(self):
        """Carga un modelo previamente guardado"""
        try:
            self.ventana_progreso = VentanaProgreso(self.root, "Cargando Modelo")
            self.ventana_progreso.update_progress(0, "Iniciando carga del modelo...")
            
            def load_model():
                try:
                    self.predictor.cargar_modelo_guardado()
                    self.root.after(0, self.after_model_load)
                except Exception as e:
                    self.root.after(0, lambda: self.show_error("Error al cargar modelo", str(e)))
                finally:
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
            
            threading.Thread(target=load_model, daemon=True).start()
            
        except Exception as e:
            self.show_error("Error", f"Error al cargar el modelo: {str(e)}")
    def export_predictions(self):
        """Exporta las predicciones a un archivo CSV"""
        try:
            # Verificar si hay predicciones para exportar
            if not self.pred_text.get(1.0, tk.END).strip():
                messagebox.showwarning("Aviso", "No hay predicciones para exportar")
                return
            
            # Abrir diálogo para guardar archivo
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
                title="Exportar Predicciones"
            )
            
            if filename:
                # Mostrar ventana de progreso
                self.ventana_progreso = VentanaProgreso(self.root, "Exportando Predicciones")
                self.ventana_progreso.update_progress(0, "Preparando datos...")
                
                def export():
                    try:
                        # Obtener las últimas predicciones
                        predicciones = self.predictor.predecir_proximo_periodo(self.dataset)
                        
                        # Crear DataFrame con las predicciones
                        data = []
                        for pred in predicciones:
                            data.append({
                                'Fecha': pred['fecha'],
                                'Hora': pred['hora'],
                                'Categoría': pred['categoria'],
                                'Temperatura': f"{pred['temperatura']:.1f}°C",
                                'Confianza': f"{pred['confianza']*100:.1f}%",
                                'Descripción': pred['detalles']['descripcion'],
                                'Recomendaciones': ', '.join(pred['detalles']['recomendaciones'])
                            })
                        
                        # Crear y guardar DataFrame
                        df = pd.DataFrame(data)
                        df.to_csv(filename, index=False, encoding='utf-8')
                        
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Éxito", 
                            "Predicciones exportadas exitosamente"
                        ))
                        
                    except Exception as e:
                        self.root.after(0, lambda: self.show_error(
                            "Error al exportar", 
                            str(e)
                        ))
                    finally:
                        if self.ventana_progreso:
                            self.ventana_progreso.safe_destroy()
                
                # Ejecutar exportación en un hilo separado
                threading.Thread(target=export, daemon=True).start()
                
        except Exception as e:
            self.show_error("Error", f"Error al exportar predicciones: {str(e)}")        


    def after_model_load(self):
        """Acciones después de cargar el modelo"""
        self.train_log.delete(1.0, tk.END)
        self.train_log.insert(tk.END, "Modelo cargado exitosamente\n")
        self.predict_button.config(state='normal')
        messagebox.showinfo("Éxito", "Modelo cargado exitosamente")
    def update_training_progress(self, epoch, total_epochs):
        """Actualiza el progreso del entrenamiento en la interfaz"""
        try:
            if self.ventana_progreso and self.ventana_progreso.winfo_exists():
                # Calcular el progreso actual
                progress = ((epoch + 1) / total_epochs) * 100
                
                # Actualizar la ventana de progreso
                self.ventana_progreso.update_progress(
                    progress,
                    f"Entrenando época {epoch + 1}/{total_epochs}"
                )
                
                # Actualizar el log de entrenamiento
                self.root.after(0, lambda: self.train_log.insert(tk.END, 
                    f"\nProgreso: {progress:.1f}% - Época {epoch + 1}/{total_epochs}"))
                self.root.after(0, lambda: self.train_log.see(tk.END))
                
                # Forzar actualización de la interfaz
                self.root.update_idletasks()
        except Exception as e:
            print(f"Error actualizando progreso: {str(e)}")
            if self.ventana_progreso:
                try:
                    self.ventana_progreso.safe_destroy()
                except:
                    pass
    def load_dataset(self):
        """Carga el dataset seleccionado"""
        try:
            filename = filedialog.askopenfilename(
                title="Seleccionar Dataset",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if filename:
                self.file_label.config(text=os.path.basename(filename))
                self.ventana_progreso = VentanaProgreso(self.root, "Cargando Dataset")
                self.ventana_progreso.update_progress(0, "Iniciando carga...")
                
                def load_data():
                    try:
                        self.dataset = self.predictor.cargar_datos(filename)
                        self.dataset_path = filename  # Guarda la ruta para uso futuro
                        self.root.after(0, self.after_dataset_load)
                    except Exception as e:
                        error_msg = str(e)  # Captura el valor inmediatamente
                        self.root.after(0, lambda: self.show_error("Error al cargar datos", error_msg))
                    finally:
                        # Usar método safe_destroy en lugar de acceder a tk directamente
                        if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                            self.ventana_progreso.safe_destroy()
                
                threading.Thread(target=load_data, daemon=True).start()
                
        except Exception as e:
            self.show_error("Error", f"Error al seleccionar archivo: {str(e)}")
    
    def after_dataset_load(self):
        """Acciones después de cargar el dataset"""
        self.show_dataset_info()
        self.enable_training_controls()
        self.predict_button.config(state='normal')
        
    def show_dataset_info(self):
        """Muestra información del dataset y visualizaciones"""
        try:
            # Limpiar y mostrar información básica
            self.data_text.delete(1.0, tk.END)
            info = f"Registros totales: {len(self.dataset)}\n"
            info += f"Rango de fechas: {str(self.dataset.index.min())} a {str(self.dataset.index.max())}\n\n"
            
            # Añadir resumen de variables
            info += "Resumen de variables:\n"
            for column in self.dataset.columns:
                info += f"\n{column}:\n"
                try:
                    info += f"  Min: {self.dataset[column].min()}\n"
                    info += f"  Max: {self.dataset[column].max()}\n"
                    info += f"  Media: {self.dataset[column].mean()}\n"
                    info += f"  Desv. Est.: {self.dataset[column].std()}\n"
                    info += f"  Valores nulos: {self.dataset[column].isnull().sum()}\n"
                except:
                    info += "  No se pueden calcular estadísticas para esta variable\n"
            
            # Añadir información de correlaciones
            info += "\nMatriz de correlación:\n"
            try:
                correlaciones = self.dataset.corr().round(2)
                info += str(correlaciones)
            except:
                info += "No se pueden calcular las correlaciones\n"
            
            info += "\nEstadísticas descriptivas completas:\n"
            try:
                desc = self.dataset.describe().round(2)
                info += str(desc)
            except:
                info += "No se pueden calcular las estadísticas descriptivas completas\n"
            
            self.data_text.insert(tk.END, info)

            # Crear visualización de series temporales
            try:
                # Inicializar predictor si no existe
                if not hasattr(self, 'predictor'):
                    self.predictor = PrediccionMeteo()
                    
                # Crear visualizador y asignarlo al predictor
                self.predictor.visualizador = VisualizacionMicroclima()
                
                # Mostrar ventana de progreso para visualizaciones
                self.ventana_progreso = VentanaProgreso(self.root, "Generando Visualizaciones")
                self.ventana_progreso.update_progress(10, "Preparando gráficos...")
                
                def create_visualizations():
                    try:
                        # Crear las figuras en segundo plano
                        self.ventana_progreso.update_progress(30, "Generando series temporales...")
                        fig_series = self.predictor.visualizador.plot_series_temporal(self.dataset)
                        
                        self.ventana_progreso.update_progress(60, "Generando distribuciones...")
                        fig_dist = self.predictor.visualizador.plot_distribucion_condiciones(self.dataset)
                        
                        # Crear directorio temporal si no existe
                        if not os.path.exists('temp_figures'):
                            os.makedirs('temp_figures')
                        
                        # Guardar figuras temporalmente
                        try:
                            fig_series.savefig('temp_figures/series_temporal.png')
                            fig_dist.savefig('temp_figures/distribucion.png')
                        except Exception as e:
                            print(f"Error al guardar figuras: {e}")
                        
                        # Actualizar la interfaz en el hilo principal
                        def update_ui():
                            try:
                                # Asegurarse de que la ventana de visualización exista
                                if not hasattr(self, 'ventana_viz') or self.ventana_viz is None or not self.ventana_viz.winfo_exists():
                                    self.ventana_viz = VentanaVisualizacion(self.root)
                                    
                                    # Configurar tamaño inicial adecuado
                                    screen_width = self.root.winfo_screenwidth()
                                    screen_height = self.root.winfo_screenheight()
                                    width = int(min(screen_width * 0.8, 1000))
                                    height = int(min(screen_height * 0.8, 600))
                                    self.ventana_viz.geometry(f"{width}x{height}")
                                    
                                    # Centrar la ventana
                                    x = (screen_width - width) // 2
                                    y = (screen_height - height) // 2
                                    self.ventana_viz.geometry(f"{width}x{height}+{x}+{y}")
                                
                                # Verificar que el método existe antes de llamarlo
                                if hasattr(self.ventana_viz, 'actualizar_graficas_iniciales'):
                                    self.ventana_viz.actualizar_graficas_iniciales({
                                        'series_temporal': fig_series,
                                        'distribucion': fig_dist
                                    })
                                else:
                                    print("ERROR: Método 'actualizar_graficas_iniciales' no encontrado")
                                
                                # Traer ventana al frente
                                self.ventana_viz.lift()
                                
                                # Habilitar controles relevantes
                                self.enable_training_controls()
                                
                                # Cerrar ventana de progreso
                                if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                                    self.ventana_progreso.safe_destroy()
                                    
                            except Exception as ui_error:
                                print(f"Error actualizando UI: {ui_error}")
                                import traceback
                                traceback.print_exc()
                                
                                # Cerrar ventana de progreso
                                if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                                    self.ventana_progreso.safe_destroy()
                        
                        # Actualizar UI desde el hilo principal
                        self.root.after(0, update_ui)
                        
                        # Limpiar memoria de las figuras cuando ya no se necesiten
                        self.root.after(1000, lambda: plt.close(fig_series))
                        self.root.after(1000, lambda: plt.close(fig_dist))
                        
                    except Exception as vis_error:
                        # Manejar errores y cerrar ventana de progreso
                        error_msg = str(vis_error)
                        print(f"Error en visualización: {error_msg}")
                        
                        def show_vis_error():
                            self.show_error("Error en visualización", 
                                        f"Error al mostrar visualizaciones: {error_msg}")
                            if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                                self.ventana_progreso.safe_destroy()
                        
                        self.root.after(0, show_vis_error)
                
                # Ejecutar creación de visualizaciones en un hilo separado
                threading.Thread(target=create_visualizations, daemon=True).start()
                    
            except Exception as e:
                self.show_error("Error", 
                            f"Error al mostrar información del dataset: {str(e)}")
                print(f"Error detallado: {str(e)}")
                
        except Exception as e:
            self.show_error("Error", 
                        f"Error al mostrar información del dataset: {str(e)}")
            print(f"Error detallado: {str(e)}")
            
        finally:
            # Limpiar archivos temporales antiguos
            try:
                if os.path.exists('temp_figures'):
                    for file in os.listdir('temp_figures'):
                        if file.endswith('.png'):
                            file_path = os.path.join('temp_figures', file)
                            if os.path.getctime(file_path) < time.time() - 3600:  # Más de 1 hora
                                os.remove(file_path)
            except Exception as e:
                print(f"Error al limpiar archivos temporales: {e}")
            
            # Asegurar que el cursor vuelva al inicio del texto
            self.data_text.see("1.0")
            
            # Actualizar la interfaz
            self.root.update_idletasks()
            
            # Limpiar memoria
            gc.collect()
    def train_model(self):
        """Inicia el entrenamiento del modelo con los parámetros configurados"""
        if not hasattr(self, 'dataset'):
            self.show_error("Error", "Primero debe cargar un dataset")
            return
                
        try:
            # Obtener valores de los parámetros
            try:
                epochs = int(self.epochs_var.get())
                batch_size = int(self.batch_size_var.get())
                learning_rate = float(self.learning_rate_var.get())
                lstm_units = int(self.lstm_units_var.get())
                dropout_rate = float(self.dropout_var.get())
                l2_reg = float(self.l2_reg_var.get())
                use_ensemble = self.use_ensemble_var.get()
            except ValueError as e:
                self.show_error("Error de Entrada", 
                            "Por favor, ingrese valores numéricos válidos para todos los parámetros.")
                return
            
            # Verificar rangos válidos
            if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or lstm_units <= 0:
                self.show_error("Error de Entrada", 
                            "Los valores de épocas, batch size, learning rate y unidades LSTM deben ser positivos.")
                return
                
            if dropout_rate < 0 or dropout_rate > 0.9:
                self.show_error("Error de Entrada", 
                            "La tasa de dropout debe estar entre 0 y 0.9.")
                return
                
            # Actualizar los parámetros del predictor con los valores configurados
            self.predictor.LEARNING_RATE = learning_rate
            self.predictor.BATCH_SIZE = batch_size
            self.predictor.use_ensemble = use_ensemble
            
            # Limpiar el log y mostrar configuración
            self.train_log.delete(1.0, tk.END)
            self.train_log.insert(tk.END, "Iniciando entrenamiento con la siguiente configuración:\n")
            self.train_log.insert(tk.END, f"- Épocas: {epochs}\n")
            self.train_log.insert(tk.END, f"- Batch Size: {batch_size}\n")
            self.train_log.insert(tk.END, f"- Learning Rate: {learning_rate}\n")
            self.train_log.insert(tk.END, f"- Unidades LSTM: {lstm_units}\n")
            self.train_log.insert(tk.END, f"- Dropout: {dropout_rate}\n")
            self.train_log.insert(tk.END, f"- Regularización L2: {l2_reg}\n")
            self.train_log.insert(tk.END, f"- Usar Ensemble: {'Sí' if use_ensemble else 'No'}\n\n")
            
            # Mostrar ventana de progreso
            self.ventana_progreso = VentanaProgreso(self.root, "Entrenando Modelo")
            self.entrenamiento_activo = True

            # Variables para controlar el progreso por fases
            self.fase_actual = 1
            self.total_fases = 2
            self.max_epochs_por_fase = {1: 25, 2: epochs}

            # Callback para manejar fases
            def progress_callback(epoch, max_epochs, fase=None, total_fases=None):
                """Callback que maneja correctamente las distintas fases de entrenamiento"""
                if self.entrenamiento_activo and hasattr(self, 'ventana_progreso'):
                    try:
                        # Actualizar fase si se proporciona
                        if fase is not None:
                            self.fase_actual = fase
                        if total_fases is not None:
                            self.total_fases = total_fases
                        
                        # Calcular progreso considerando la fase actual
                        peso_por_fase = 100.0 / self.total_fases
                        progreso_fase = ((epoch + 1) / max_epochs) * peso_por_fase
                        progreso_anterior = (self.fase_actual - 1) * peso_por_fase
                        progreso_total = min(progreso_anterior + progreso_fase, 100.0)  # Asegurar máximo 100%
                        
                        # Actualizar la ventana de progreso
                        mensaje = f"Fase {self.fase_actual}/{self.total_fases} - Época {epoch + 1}/{max_epochs}"
                        self.ventana_progreso.update_progress(progreso_total, mensaje)
                        
                        # Actualizar log
                        self.root.after(0, lambda: self.train_log.insert(tk.END, 
                            f"\n{mensaje} - Progreso: {progreso_total:.1f}%"))
                        self.root.after(0, lambda: self.train_log.see(tk.END))
                        
                        # Forzar actualización
                        self.root.update_idletasks()
                    except Exception as e:
                        print(f"Error en callback de progreso: {e}")
            
            def handle_error(error_msg):
                """Maneja los errores de entrenamiento"""
                def show_error():
                    self.show_error("Error en entrenamiento", error_msg)
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
                self.root.after(0, show_error)

            def handle_success(history):
                """Maneja el éxito del entrenamiento"""
                def update_ui():
                    self.update_training_results(history)
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
                    
                    # Mostrar mensaje de éxito
                    messagebox.showinfo("Éxito", "Entrenamiento completado exitosamente")
                    
                    # Habilitar botones de predicción
                    self.predict_button.config(state='normal')
                    
                self.root.after(0, update_ui)

            def train():
                try:
                    print("Iniciando hilo de entrenamiento...")
                    print("Preparando datos para entrenamiento...")
                    print(f"Iniciando entrenamiento con {epochs} épocas y batch size de {batch_size}")
                    
                    # Configurar parámetros adicionales del modelo
                    self.predictor.create_model_params = {
                        'lstm_units': lstm_units,
                        'dropout_rate': dropout_rate,
                        'l2_reg': l2_reg
                    }
                    
                    # Enviar el nuevo callback al método de entrenamiento
                    history = self.predictor.entrenar_modelo(
                        df=self.dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callback=progress_callback,
                        learning_rate=learning_rate  # Añadir learning rate
                    )
                    
                    handle_success(history)
                    
                except Exception as e:
                    handle_error(str(e))
                finally:
                    self.entrenamiento_activo = False
                    print("Entrenamiento finalizado")
            
            # Iniciar entrenamiento en un hilo separado
            training_thread = threading.Thread(target=train, daemon=True)
            training_thread.start()
            
        except ValueError:
            self.show_error("Error", "Los valores de épocas y batch size deben ser números enteros")
        except Exception as e:
            self.show_error("Error", f"Error al iniciar entrenamiento: {str(e)}")
            if self.ventana_progreso:
                self.ventana_progreso.safe_destroy()
    def update_training_results(self, history):
        """Actualiza los resultados del entrenamiento en la interfaz"""
        try:
            # Limpiar log anterior
            self.train_log.delete(1.0, tk.END)
            self.train_log.insert(tk.END, "Entrenamiento completado\n\n")
            
            # Verificar si history es None
            if history is None:
                self.train_log.insert(tk.END, "No hay métricas disponibles del entrenamiento\n")
                self.predict_button.config(state='disabled')
                messagebox.showwarning("Advertencia", "El entrenamiento no generó métricas")
                return
            
            # Verificar si history tiene el atributo history
            if not hasattr(history, 'history'):
                self.train_log.insert(tk.END, "No se obtuvieron métricas del entrenamiento\n")
                self.predict_button.config(state='disabled')
                messagebox.showwarning("Advertencia", "No se obtuvieron métricas del entrenamiento")
                return
            
            # Mostrar métricas finales
            self.train_log.insert(tk.END, "Métricas finales:\n")
            metrics = history.history
            
            # Actualizar métricas en la interfaz
            try:
                if 'accuracy' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Precisión de entrenamiento: {metrics['accuracy'][-1]:.4f}\n")
                if 'loss' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Pérdida de entrenamiento: {metrics['loss'][-1]:.4f}\n")
                if 'val_accuracy' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Precisión de validación: {metrics['val_accuracy'][-1]:.4f}\n")
                if 'val_loss' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Pérdida de validación: {metrics['val_loss'][-1]:.4f}\n")
                
                # Métricas adicionales si están disponibles
                if 'precision' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Precisión: {metrics['precision'][-1]:.4f}\n")
                if 'recall' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Recall: {metrics['recall'][-1]:.4f}\n")
                if 'auc' in metrics:
                    self.train_log.insert(tk.END, 
                        f"AUC: {metrics['auc'][-1]:.4f}\n")
            except Exception as e:
                self.train_log.insert(tk.END, f"\nError al mostrar algunas métricas: {str(e)}\n")
            
            # Habilitar botones de predicción
            self.predict_button.config(state='normal')
            
            # Actualizar interfaz
            self.root.update_idletasks()
            
        except Exception as e:
            self.show_error("Error", f"Error al actualizar resultados: {str(e)}")
        finally:
            # Asegurar que la ventana de progreso se cierre
            if self.ventana_progreso:
                self.ventana_progreso.safe_destroy()

    def generate_predictions(self):
        """Genera predicciones con el modelo actual y abre ventanas de visualización"""
        if not hasattr(self, 'dataset'):
            self.show_error("Error", "Primero debe cargar un dataset")
            return
            
        try:
            self.ventana_progreso = VentanaProgreso(self.root, "Generando Predicciones")
            self.ventana_progreso.update_progress(10, "Iniciando predicciones...")
            
            def predict():
                try:
                    self.ventana_progreso.update_progress(30, "Procesando datos...")
                    
                    # Espera simulada para cuando el procesamiento es muy rápido
                    time.sleep(0.5)
                    
                    self.ventana_progreso.update_progress(60, "Aplicando modelo...")
                    predicciones = self.predictor.predecir_proximo_periodo(self.dataset)
                    
                    self.ventana_progreso.update_progress(90, "Finalizando y abriendo visualizaciones...")
                    self.root.after(0, lambda: self.update_predictions(predicciones))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.show_error("Error en predicción", str(e)))
                finally:
                    # Dar tiempo para ver el 100% antes de cerrar
                    self.ventana_progreso.update_progress(100, "¡Predicciones completadas! Abriendo visualizaciones...")
                    self.root.after(1000, self.ventana_progreso.safe_destroy)
            
            threading.Thread(target=predict, daemon=True).start()
            
        except Exception as e:
            self.show_error("Error", f"Error al generar predicciones: {str(e)}")
            if hasattr(self, 'ventana_progreso') and self.ventana_progreso:
                self.ventana_progreso.safe_destroy()
            
    def update_predictions(self, predicciones):
        """Actualiza la interfaz con las nuevas predicciones y abre ventanas de visualización"""
        try:
            # Actualizar información actual
            primera_pred = predicciones[0]
            
            # Verificar si current_weather_frame existe
            if hasattr(self, 'current_weather_frame'):
                # Limpiar el frame actual
                for widget in self.current_weather_frame.winfo_children():
                    widget.destroy()
                
                # Crear un nuevo display de temperatura con valores actualizados
                self.temp_display = create_temperature_display(
                    self.current_weather_frame,
                    f"{primera_pred['temperatura']:.1f}°C",
                    f"{primera_pred['confianza']*100:.1f}%",
                    self.colors
                )
            else:
                # Usar el método tradicional si no existe current_weather_frame
                if hasattr(self, 'current_temp_label'):
                    self.current_temp_label.config(text=f"{primera_pred['temperatura']:.1f}°C")
                if hasattr(self, 'current_condition_label'):
                    self.current_condition_label.config(text=primera_pred['categoria'])
                if hasattr(self, 'current_confidence_label'):
                    self.current_confidence_label.config(text=f"Confianza: {primera_pred['confianza']*100:.1f}%")
            
            # Actualizar texto de predicciones
            self.pred_text.delete(1.0, tk.END)
            for pred in predicciones:
                self.pred_text.insert(tk.END, 
                    f"{pred['fecha']}: {pred['categoria']} "
                    f"(Confianza: {pred['confianza']*100:.1f}%)\n"
                    f"Temperatura: {pred['temperatura']:.1f}°C\n"
                    f"{pred['detalles']['descripcion']}\n"
                    f"Recomendaciones: {', '.join(pred['detalles']['recomendaciones'])}\n\n")
            
            # === NUEVO CÓDIGO: ABRIR VENTANAS AUTOMÁTICAMENTE ===
            
            # Mostrar ventanas solo si no están ya abiertas
            # 1. Abrir ventana de temperatura
            if not hasattr(self, 'ventana_viz') or not self.ventana_viz or not self.ventana_viz.winfo_exists():
                # Crear la ventana de visualización de temperatura
                self.root.after(100, lambda: self.mostrar_visualizaciones("temperatura"))
            else:
                # Si ya existe, actualizar con nuevas predicciones
                self.ventana_viz.actualizar_grafica(predicciones, self.visualizador)
                self.ventana_viz.lift()  # Traer al frente
            
            # 2. Abrir ventana de pronóstico detallado
            # Retrasar ligeramente para evitar conflictos de interfaz
            def abrir_pronostico_detallado():
                if not hasattr(self, 'ventana_pronostico') or not self.ventana_pronostico or not self.ventana_pronostico.winfo_exists():
                    # Crear la ventana de pronóstico detallado
                    self.mostrar_visualizaciones("detallado")
                else:
                    # Si ya existe, actualizar con nuevas predicciones
                    self.ventana_pronostico.actualizar_grafica(predicciones, self.visualizador)
                    self.ventana_pronostico.lift()  # Traer al frente
                    
            # Programar la apertura después de un breve retraso
            self.root.after(500, abrir_pronostico_detallado)
            
            # === FIN DEL NUEVO CÓDIGO ===
                
            print(f"Temperatura actualizada a: {primera_pred['temperatura']:.1f}°C")
            
        except Exception as e:
            import traceback
            print(f"Error al actualizar predicciones: {e}")
            traceback.print_exc()

    # === MEJORA ADICIONAL: MODIFICAR EL MÉTODO GENERATE_PREDICTIONS ===
    # Para mostrar un mensaje que indique que se abrirán las ventanas
            
    def show_error(self, title, message):
        """Muestra un mensaje de error"""
        messagebox.showerror(title, message)
        
    def enable_training_controls(self):
        """Habilita los controles de entrenamiento"""
        for child in self.scrollable_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state='normal')
    ## Integracion interfaz del programa de Maquina Del Tiempo :3
    def create_station_tab(self, parent):
        """Crea una nueva pestaña para la estación meteorológica"""
        station_frame = ttk.LabelFrame(parent, text="Estación Meteorológica", padding="5")
        station_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        
        # Botón para abrir la ventana de la estación meteorológica
        ttk.Button(station_frame, 
                text="Abrir Estación Meteorológica",
                command=self.open_station_window, 
                width=28).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Nuevo botón para procesar archivos de estación
        ttk.Button(station_frame, 
                text="Procesar Archivo de Estación",
                command=self.open_process_window, 
                width=25).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Estado de conexión
        self.station_status_label = ttk.Label(station_frame, 
                                        text="Estado: No conectado", 
                                        style='Info.TLabel')
        self.station_status_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Botón para importar datos de estación
        ttk.Button(station_frame, 
                text="Importar Datos de Estación",
                command=self.import_station_data, 
                width=25).pack(side=tk.LEFT, padx=5, pady=5)

    def open_station_window(self):
        """Abre la ventana de la estación meteorológica"""
        try:
            # Crear una nueva ventana de nivel superior
            station_window = tk.Toplevel(self.root)
            station_window.title("Estación Meteorológica - Sistema de Monitoreo")
            
            # Configurar la ventana
            station_window.geometry("1200x700")
            station_window.minsize(1000, 600)
            
            # Crear aplicación de estación meteorológica
            self.station_app = EstacionMeteoApp(station_window)
            
            # Vincular eventos de cierre
            station_window.protocol("WM_DELETE_WINDOW", lambda: self.on_station_window_close(station_window))
            
            # Actualizar etiqueta de estado
            self.station_status_label.config(text="Estado: Ventana abierta")
            
        except Exception as e:
            self.show_error("Error", f"Error al abrir la estación meteorológica: {str(e)}")

    def open_process_window(self):
        """Abre la ventana para procesar archivos de la estación meteorológica"""
        try:
            # Importar la clase EstacionMeteorologicaGUI de ProcesarDatosGUI.py
            from ProcesarDatosGUI import EstacionMeteorologicaGUI
            
            # Crear una nueva ventana de nivel superior
            process_window = tk.Toplevel(self.root)
            process_window.title("Procesador de Datos Meteorológicos")
            
            # Configurar la ventana
            process_window.geometry("800x600")
            process_window.minsize(700, 500)
            
            # Crear instancia de la aplicación
            self.process_app = EstacionMeteorologicaGUI(process_window)
            
            # Configurar la ventana para que se destruya al cerrarse
            process_window.protocol("WM_DELETE_WINDOW", lambda: self.on_process_window_close(process_window))
            
            # Actualizar etiqueta de estado
            self.station_status_label.config(text="Estado: Procesando datos")
            
        except Exception as e:
            self.show_error("Error", f"Error al abrir el procesador de datos: {str(e)}")

    def on_process_window_close(self, window):
        """Maneja el cierre de la ventana de procesamiento"""
        try:
            # Actualizar estado
            self.station_status_label.config(text="Estado: Procesamiento cerrado")
            
            # Destruir ventana
            window.destroy()
            
        except Exception as e:
            print(f"Error al cerrar ventana de procesamiento: {str(e)}")
    def on_station_window_close(self, window):
        """Maneja el cierre de la ventana de estación"""
        try:
            # Actualizar estado
            self.station_status_label.config(text="Estado: Ventana cerrada")
            
            # Destruir ventana
            window.destroy()
            
        except Exception as e:
            print(f"Error al cerrar ventana: {str(e)}")

    def import_station_data(self):
        """Importa datos desde la estación meteorológica al sistema de predicción"""
        try:
            # Verificar si la aplicación de estación está activa
            if not hasattr(self, 'station_app'):
                messagebox.showinfo("Información", "Primero debe abrir la ventana de la Estación Meteorológica")
                return
            
            # Verificar si hay datos cargados
            if not hasattr(self.station_app, 'data') or not self.station_app.data['fecha']:
                messagebox.showinfo("Información", "No hay datos cargados en la Estación Meteorológica")
                return
            
            # Mostrar ventana de progreso
            self.ventana_progreso = VentanaProgreso(self.root, "Importando Datos")
            self.ventana_progreso.update_progress(10, "Preparando datos de la estación...")
            
            def import_data_task():
                try:
                    # Convertir datos de la estación a formato compatible con el predictor
                    self.ventana_progreso.update_progress(30, "Procesando formato de datos...")
                    
                    # Crear DataFrame a partir de los datos de la estación
                    import pandas as pd
                    from datetime import datetime
                    
                    data_dict = {}
                    for key in self.station_app.data:
                        if len(self.station_app.data[key]) > 0:
                            data_dict[key] = self.station_app.data[key]
                    
                    # Verificar si hay datos suficientes
                    if not data_dict or 'fecha' not in data_dict:
                        raise ValueError("No hay datos suficientes en la estación para importar")
                    
                    # Crear DataFrame
                    df_estacion = pd.DataFrame(data_dict)
                    
                    # Convertir fechas a datetime si son strings
                    if isinstance(df_estacion['fecha'][0], str):
                        df_estacion['fecha'] = pd.to_datetime(df_estacion['fecha'])
                    
                    # Establecer fecha como índice
                    df_estacion.set_index('fecha', inplace=True)
                    
                    # Guardar datos temporalmente
                    temp_file = "temp_station_data.csv"
                    df_estacion.to_csv(temp_file)
                    
                    self.ventana_progreso.update_progress(60, "Cargando datos en el sistema de predicción...")
                    
                    # Cargar datos en el sistema de predicción
                    try:
                        self.dataset = self.predictor.cargar_datos(temp_file)
                        self.dataset_path = temp_file
                    except Exception as e:
                        raise ValueError(f"Error al cargar datos en el predictor: {str(e)}")
                    
                    self.ventana_progreso.update_progress(90, "Actualizando interfaz...")
                    
                    # Actualizar interfaz
                    self.root.after(0, self.after_dataset_load)
                    
                    # Mostrar mensaje de éxito
                    self.root.after(500, lambda: messagebox.showinfo(
                        "Éxito", 
                        f"Se importaron {len(df_estacion)} registros desde la Estación Meteorológica"
                    ))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.show_error(
                        "Error al importar", 
                        str(e)
                    ))
                finally:
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
            
            # Ejecutar tarea en segundo plano
            threading.Thread(target=import_data_task, daemon=True).start()
            
        except Exception as e:
            self.show_error("Error", f"Error al iniciar importación: {str(e)}")

if __name__ == "__main__":
    try:
        print("Iniciando aplicación...")
        root = tk.Tk()
        
        # Configurar el redimensionamiento de la ventana principal
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        app = MicroClimaGUI(root)
        print("GUI inicializada")
        
        # Centrar la ventana en la pantalla
        app.center_window()
        
        root.mainloop()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {str(e)}")

######## By: Bryan Rojas and Nathalia Gutierrez ########
# 2024-01-01
# Microclima GUI - Sistema de predicción meteorológica
# Version 1.0