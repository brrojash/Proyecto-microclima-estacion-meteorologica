import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import serial
import serial.tools.list_ports
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import re
import matplotlib.dates as mdates
import numpy as np
import math

class EstacionMeteoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estación Meteorológica - Sistema de Monitoreo Lestoma")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        # Alertas
        self.alert_update_job = None
        # Sistema de semáforo para comunicaciones
        self.is_communicating = False
        self.command_queue = []
        self.queue_processing = False
        # Configurar tema y colores
        self.configure_styles()
        
        # Variables de estado
        self.connected = False
        self.serial_conn = None
        self.password = "MaquinaDelTiempo"  # Contraseña correcta de la ESP32
        
        # Variables para datos y gráficos
        self.data = {
            'fecha': [],
            'temperatura_C': [],
            'humedad_relativa': [],
            'precipitacion_mm': [],
            'cobertura_nubes_octas': [],
            'velocidad_viento_kmh': [],
            'luminosidad_lux': [],
            'radiacion_solar_wm2': [],
            'direccion_viento': [],
            'condicion_climatica': []
        }
        
        # Variables para datos de alertas
        self.alertas_data = {
            'timestamp': [],
            'tipo_alerta': [],
            'descripcion': [],
            'volumen_l': [],
            'lluvia_mm_h': [],
            'riesgo': [],
            'humedad': []
        }
        
        # Mapeo de nombres de columnas
        self.column_mapping = {
            'fecha_hora': 'fecha',
            'temp_dht_cal': 'temperatura_C',
            'hum_dht_raw': 'humedad_relativa',
            'lluvia_mm': 'precipitacion_mm',
            'cobertura_nubes_octas': 'cobertura_nubes_octas',
            'vel_viento_kmh': 'velocidad_viento_kmh',
            'radiacion_solar_J_m2': 'luminosidad_lux',
            'radiacion_solar_wm2': 'radiacion_solar_wm2',
            'direccion_viento': 'direccion_viento',
            'condicion_climatica': 'condicion_climatica'
        }
        
        # Estado de alertas
        self.alert_active = False
        self.last_alert_check = 0
        self.alert_check_interval = 10000  # 10 segundos
        
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Pestaña de conexión
        self.conn_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.conn_tab, text="Conexión")
        
        # Pestaña de dashboard
        self.dash_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dash_tab, text="Dashboard", state="disabled")
        
        # Pestaña de cálculo de agua
        self.water_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.water_tab, text="Cálculo de Agua", state="disabled")
        
        # Nueva pestaña de alertas
        self.alerts_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.alerts_tab, text="Alertas", state="disabled")
        
        # Configurar pestaña de conexión
        self.setup_connection_tab()
        self.add_log("⚠️ Nota: Algunos caracteres especiales (ñ, tildes) pueden mostrarse incorrectamente en el log. Esto no afecta a los datos guardados.")
        # Configurar pestaña de dashboard
        self.setup_dashboard_tab()
        # Configurar pestaña de cálculo de agua
        self.setup_water_calculation_tab()
        # Configurar nueva pestaña de alertas
        self.setup_alerts_tab()
        
        # Intentar cargar caché al inicio
        loaded_cache = self.load_data_cache()
        
        # Buscar puertos al inicio
        self.add_log("Aplicación iniciada. Por favor seleccione un puerto COM y presione CONECTAR.")
        self.refresh_ports()
        
        # Si no se cargó caché, mostrar aviso
        if not loaded_cache:
            self.add_log("No se encontró caché previo o hubo un error al cargarlo.")
            
        # Programar comprobación periódica de alertas
        self.root.after(self.alert_check_interval, self.check_for_alerts)
    
    def configure_styles(self):
        """Configura estilos personalizados para la aplicación"""
        self.style = ttk.Style()
        
        # Colores
        bg_color = "#f5f5f5"
        accent_color = "#3498db"
        accent_dark = "#2980b9"
        header_color = "#2c3e50"
        success_color = "#2ecc71"
        warning_color = "#f39c12"
        error_color = "#e74c3c"
        
        # Configuración general
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10))
        self.style.configure('TEntry', font=('Segoe UI', 10))
        self.style.configure('TCombobox', font=('Segoe UI', 10))
        self.style.configure('TNotebook', background=bg_color)
        
        # Botones especiales
        self.style.configure('Big.TButton', font=('Segoe UI', 12, 'bold'))
        self.style.configure('Command.TButton', font=('Segoe UI', 10))
        
        # Etiquetas de lecturas
        self.style.configure('Reading.TLabel', font=('Segoe UI', 14, 'bold'), foreground=accent_dark)
        
        # Resultado destacado
        self.style.configure('Result.TLabel', font=('Segoe UI', 18, 'bold'), foreground=success_color)
        
        # Alertas
        self.style.configure('Alert.TLabel', font=('Segoe UI', 14, 'bold'), foreground=error_color)
        self.style.configure('Warning.TLabel', font=('Segoe UI', 12, 'bold'), foreground=warning_color)
        self.style.configure('AlertHeader.TLabel', font=('Segoe UI', 16, 'bold'), foreground=error_color)
    
    def setup_connection_tab(self):
        """Configura la pestaña de conexión"""
        main_frame = ttk.Frame(self.conn_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título de la aplicación
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(title_frame, text="Estación Meteorológica - Lestoma", 
                 font=('Segoe UI', 14, 'bold')).pack(anchor=tk.CENTER)
        
        # Sección de conexión
        conn_frame = ttk.LabelFrame(main_frame, text="Conexión a la Estación", padding=15)
        conn_frame.pack(fill=tk.X, pady=10)
        
        # Primer frame horizontal para los parámetros
        params_frame = ttk.Frame(conn_frame)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Puerto COM
        port_frame = ttk.Frame(params_frame)
        port_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(port_frame, text="Puerto COM:").pack(anchor=tk.W)
        port_select_frame = ttk.Frame(port_frame)
        port_select_frame.pack(fill=tk.X, pady=5)
        
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_select_frame, textvariable=self.port_var, width=30)
        self.port_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(port_select_frame, text="↻", width=3,
                  command=self.refresh_ports).pack(side=tk.LEFT, padx=5)
        
        # Velocidad (baudrate)
        baud_frame = ttk.Frame(params_frame)
        baud_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(baud_frame, text="Velocidad:").pack(anchor=tk.W)
        self.baudrate_var = tk.StringVar(value="115200")  # Cambiado a 115200 por defecto
        baudrate_combo = ttk.Combobox(baud_frame, textvariable=self.baudrate_var, width=10, 
                                     values=["9600", "115200"])
        baudrate_combo.pack(fill=tk.X, pady=5)
        
        # Contraseña
        pass_frame = ttk.Frame(params_frame)
        pass_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Label(pass_frame, text="Contraseña:").pack(anchor=tk.W)
        self.password_var = tk.StringVar(value=self.password)
        ttk.Entry(pass_frame, textvariable=self.password_var, width=25, show="*").pack(fill=tk.X, pady=5)
        
        # Botón de conexión y estado
        button_frame = ttk.Frame(conn_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Indicador de estado
        status_frame = ttk.Frame(button_frame)
        status_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(status_frame, text="Estado:").pack(side=tk.LEFT)
        self.status_indicator = tk.Canvas(status_frame, width=20, height=20, bg="red")
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_text = ttk.Label(status_frame, text="Desconectado")
        self.status_text.pack(side=tk.LEFT, padx=5)
        
        # Botón de conexión grande y destacado
        self.connect_btn = ttk.Button(button_frame, text="CONECTAR", 
                                     command=self.connect, width=20, style='Big.TButton')
        self.connect_btn.pack(side=tk.RIGHT, padx=10)
        
        # Área de log con barra de desplazamiento
        log_frame = ttk.LabelFrame(main_frame, text="Log de Comunicación", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=12, 
                                                  background="#000000", foreground="#00FF00", 
                                                  font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Sección para enviar comandos
        cmd_frame = ttk.LabelFrame(main_frame, text="Enviar Comandos", padding=10)
        cmd_frame.pack(fill=tk.X, pady=10)
        
        # Campo para comandos
        cmd_entry_frame = ttk.Frame(cmd_frame)
        cmd_entry_frame.pack(fill=tk.X, pady=5)
        
        self.cmd_var = tk.StringVar()
        cmd_entry = ttk.Entry(cmd_entry_frame, textvariable=self.cmd_var, width=40)
        cmd_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        cmd_entry.bind('<Return>', lambda e: self.send_command())
        
        ttk.Button(cmd_entry_frame, text="Enviar", command=self.send_command).pack(side=tk.LEFT, padx=5)
        
        # Botones para comandos comunes en una fila separada - ASEGURANDO VISIBILIDAD
        common_cmd_frame = ttk.Frame(cmd_frame)
        common_cmd_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(common_cmd_frame, text="AYUDA", style="Command.TButton",
                  command=lambda: self.send_predefined_command("AYUDA")).pack(side=tk.LEFT, padx=5)
        ttk.Button(common_cmd_frame, text="INFO", style="Command.TButton",
                  command=lambda: self.send_predefined_command("INFO")).pack(side=tk.LEFT, padx=5)
        ttk.Button(common_cmd_frame, text="DATOS", style="Command.TButton",
                  command=lambda: self.send_predefined_command("DATOS")).pack(side=tk.LEFT, padx=5)
        ttk.Button(common_cmd_frame, text="ALERTAS", style="Command.TButton",
                  command=lambda: self.send_predefined_command("ALERTAS")).pack(side=tk.LEFT, padx=5)
        ttk.Button(common_cmd_frame, text="SET FECHA/HORA", style="Command.TButton",
                  command=self.set_datetime).pack(side=tk.LEFT, padx=5)

    def setup_dashboard_tab(self):
        """Configura la pestaña de dashboard con gráficos"""
        # Frame principal
        main_frame = ttk.Frame(self.dash_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cabecera con título
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(header_frame, text="Dashboard de Monitoreo - Estación Lestoma", 
                font=('Segoe UI', 14, 'bold')).pack(anchor=tk.CENTER)
        
        # Panel de control superior
        control_frame = ttk.LabelFrame(main_frame, text="Control de Datos", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Botones principales
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(buttons_frame, text="Actualizar Datos", 
                command=lambda: self.send_predefined_command("DATOS"), 
                style="Command.TButton").pack(side=tk.LEFT, padx=5)
        # Se implemento en PROCESARGUI
        # ttk.Button(buttons_frame, text="Datos por Rango", 
        #         command=self.download_data_by_range, 
        #         style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="Cargar Caché", 
                command=self.load_data_cache, 
                style="Command.TButton").pack(side=tk.LEFT, padx=5)

        ttk.Button(buttons_frame, text="Guardar Caché", 
                command=self.save_data_cache, 
                style="Command.TButton").pack(side=tk.LEFT, padx=5)
                
        ttk.Button(buttons_frame, text="Guardar Original", 
                command=self.export_original_data).pack(side=tk.LEFT, padx=5)
          # Se implemento en PROCESARGUI
        # ttk.Button(buttons_frame, text="Guardar Procesado", 
        #         command=self.export_processed_data).pack(side=tk.LEFT, padx=5)
        
        # Controles de actualización automática
        auto_frame = ttk.Frame(control_frame)
        auto_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(auto_frame, text="Intervalo de actualización:").pack(side=tk.LEFT, padx=(20, 5))
        self.update_interval = tk.StringVar(value="30")
        interval_combo = ttk.Combobox(auto_frame, textvariable=self.update_interval, width=5,
                                    values=["10", "30", "60", "300", "600"])
        interval_combo.pack(side=tk.LEFT)
        ttk.Label(auto_frame, text="segundos").pack(side=tk.LEFT, padx=5)
        
        self.auto_update_var = tk.BooleanVar(value=False)
        auto_check = ttk.Checkbutton(auto_frame, text="Actualización automática", 
                                    variable=self.auto_update_var, command=self.toggle_auto_update)
        auto_check.pack(side=tk.LEFT, padx=20)
        
        # NUEVO: Controles de actualización de alertas
        alerts_control_frame = ttk.Frame(control_frame)
        alerts_control_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(alerts_control_frame, text="Actualización alertas:").pack(side=tk.LEFT, padx=(20, 5))
        self.alerts_auto_update_var = tk.BooleanVar(value=True)
        alerts_check = ttk.Checkbutton(alerts_control_frame, text="Activa", 
                                    variable=self.alerts_auto_update_var, 
                                    command=self.toggle_alerts_auto_update)
        alerts_check.pack(side=tk.LEFT, padx=5)
        
        # Botón para forzar actualización manual
        ttk.Button(alerts_control_frame, text="Forzar actualización alertas", 
                command=lambda: self.send_predefined_command("ALERTAS")).pack(side=tk.LEFT, padx=20)
        
        # Indicador de alerta en el dashboard
        self.alert_indicator_frame = ttk.Frame(control_frame)
        self.alert_indicator_frame.pack(side=tk.RIGHT, padx=20)
        
        self.alert_indicator = tk.Canvas(self.alert_indicator_frame, width=15, height=15, bg="#cccccc")
        self.alert_indicator.pack(side=tk.LEFT, padx=5)
        
        self.alert_status_label = ttk.Label(self.alert_indicator_frame, text="Sin alertas")
        self.alert_status_label.pack(side=tk.LEFT, padx=5)
        
        # Panel de últimas lecturas (reducido)
        readings_frame = ttk.LabelFrame(main_frame, text="Últimas Lecturas", padding=5)
        readings_frame.pack(fill=tk.X, pady=5)
        
        # Crear widgets para mostrar las lecturas actuales (3 filas de 3 columnas)
        self.reading_labels = {}
        self.reading_values = {}
        
        readings_row1 = ttk.Frame(readings_frame)
        readings_row1.pack(fill=tk.X, pady=2)
        
        # Temperatura
        temp_frame = ttk.Frame(readings_row1, padding=2)
        temp_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(temp_frame, text="Temperatura:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['temperatura_C'] = ttk.Label(temp_frame, text="--.- °C", style='Reading.TLabel')
        self.reading_values['temperatura_C'].pack(side=tk.LEFT, padx=5)
        
        # Humedad
        hum_frame = ttk.Frame(readings_row1, padding=2)
        hum_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(hum_frame, text="Humedad:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['humedad_relativa'] = ttk.Label(hum_frame, text="--.- %", style='Reading.TLabel')
        self.reading_values['humedad_relativa'].pack(side=tk.LEFT, padx=5)
        
        # Precipitación
        rain_frame = ttk.Frame(readings_row1, padding=2)
        rain_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(rain_frame, text="Precipitación:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['precipitacion_mm'] = ttk.Label(rain_frame, text="--.- mm", style='Reading.TLabel')
        self.reading_values['precipitacion_mm'].pack(side=tk.LEFT, padx=5)
        
        readings_row2 = ttk.Frame(readings_frame)
        readings_row2.pack(fill=tk.X, pady=2)
        
        # Velocidad de viento
        wind_frame = ttk.Frame(readings_row2, padding=2)
        wind_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(wind_frame, text="Velocidad Viento:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['velocidad_viento_kmh'] = ttk.Label(wind_frame, text="--.- km/h", style='Reading.TLabel')
        self.reading_values['velocidad_viento_kmh'].pack(side=tk.LEFT, padx=5)
        
        # Dirección viento
        dir_frame = ttk.Frame(readings_row2, padding=2)
        dir_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(dir_frame, text="Dirección Viento:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['direccion_viento'] = ttk.Label(dir_frame, text="---", style='Reading.TLabel')
        self.reading_values['direccion_viento'].pack(side=tk.LEFT, padx=5)
        
        # Luminosidad
        lux_frame = ttk.Frame(readings_row2, padding=2)
        lux_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(lux_frame, text="Luminosidad:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['luminosidad_lux'] = ttk.Label(lux_frame, text="--.- lux", style='Reading.TLabel')
        self.reading_values['luminosidad_lux'].pack(side=tk.LEFT, padx=5)
        
        readings_row3 = ttk.Frame(readings_frame)
        readings_row3.pack(fill=tk.X, pady=2)
        
        # Radiación solar
        rad_frame = ttk.Frame(readings_row3, padding=2)
        rad_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(rad_frame, text="Radiación Solar:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['radiacion_solar_wm2'] = ttk.Label(rad_frame, text="--.- W/m²", style='Reading.TLabel')
        self.reading_values['radiacion_solar_wm2'].pack(side=tk.LEFT, padx=5)
        
        # Cobertura nubes
        clouds_frame = ttk.Frame(readings_row3, padding=2)
        clouds_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(clouds_frame, text="Cobertura Nubes:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['cobertura_nubes_octas'] = ttk.Label(clouds_frame, text="- octas", style='Reading.TLabel')
        self.reading_values['cobertura_nubes_octas'].pack(side=tk.LEFT, padx=5)
        
        # Condición climática
        cond_frame = ttk.Frame(readings_row3, padding=2)
        cond_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(cond_frame, text="Condición:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.reading_values['condicion_climatica'] = ttk.Label(cond_frame, text="---", style='Reading.TLabel')
        self.reading_values['condicion_climatica'].pack(side=tk.LEFT, padx=5)
        
        # Frame para gráficos (asegurando que ocupe más espacio)
        self.charts_frame = ttk.LabelFrame(main_frame, text="Gráficos", padding=5)
        self.charts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Crear pestañas para los gráficos
        self.charts_notebook = ttk.Notebook(self.charts_frame)
        self.charts_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña para temperatura y humedad
        self.temp_hum_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.temp_hum_tab, text="Temp / Humedad")
        
        # Pestaña para precipitación
        self.rain_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.rain_tab, text="Precipitación")
        
        # Pestaña para viento
        self.wind_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.wind_tab, text="Viento")
        
        # Pestaña para radiación/luminosidad
        self.rad_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.rad_tab, text="Radiación/Luz")
        
        # Inicializar gráficos vacíos
        self.init_charts()
        
        # Variable para control de actualización automática
        self.auto_update_job = None
    def toggle_alerts_auto_update(self):
        """Activa o desactiva la actualización automática de alertas"""
        if self.alerts_auto_update_var.get():
            # Activar actualización automática
            self.resume_automatic_updates()
            self.add_log("Actualización automática de alertas activada")
        else:
            # Desactivar actualización automática
            if self.alert_update_job:
                self.root.after_cancel(self.alert_update_job)
                self.alert_update_job = None
            self.add_log("Actualización automática de alertas desactivada")

    def setup_alerts_tab(self):
        """Configura la nueva pestaña de alertas"""
        main_frame = ttk.Frame(self.alerts_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título y control de alertas
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        # Título con indicador de estado
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.alerts_title = ttk.Label(title_frame, text="Sistema de Alertas - Sin alertas activas", 
                                    font=('Segoe UI', 16, 'bold'))
        self.alerts_title.pack(anchor=tk.W)
        
        # Botones de control
        controls_frame = ttk.Frame(header_frame)
        controls_frame.pack(side=tk.RIGHT)
        
        ttk.Button(controls_frame, text="Actualizar Alertas", 
                  command=lambda: self.send_predefined_command("ALERTAS"),
                  style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Exportar Alertas", 
                  command=self.export_alerts,
                  style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        # Panel superior: Estado actual y resumen
        status_frame = ttk.LabelFrame(main_frame, text="Estado Actual del Sistema", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Dividir en dos columnas
        left_status = ttk.Frame(status_frame)
        left_status.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        right_status = ttk.Frame(status_frame)
        right_status.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
        
        # Columna izquierda: Indicadores de estado
        indicator_frame = ttk.Frame(left_status)
        indicator_frame.pack(fill=tk.X, pady=5)
        
        # Estado de alerta
        alert_state_frame = ttk.Frame(indicator_frame)
        alert_state_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(alert_state_frame, text="Estado de Alerta:", 
                 font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=5)
        
        self.alert_state_canvas = tk.Canvas(alert_state_frame, width=20, height=20, bg="#cccccc")
        self.alert_state_canvas.pack(side=tk.LEFT, padx=5)
        
        self.alert_state_text = ttk.Label(alert_state_frame, text="Sin alertas activas", 
                                        font=('Segoe UI', 11, 'bold'))
        self.alert_state_text.pack(side=tk.LEFT, padx=5)
        
        # Último nivel de riesgo
        risk_frame = ttk.Frame(indicator_frame)
        risk_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(risk_frame, text="Último nivel de riesgo:", 
                 font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=5)
        
        self.risk_level_text = ttk.Label(risk_frame, text="0.0%", 
                                       font=('Segoe UI', 11, 'bold'))
        self.risk_level_text.pack(side=tk.LEFT, padx=5)
        
        # Volumen de agua
        volume_frame = ttk.Frame(indicator_frame)
        volume_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(volume_frame, text="Volumen actual:", 
                 font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=5)
        
        self.volume_text = ttk.Label(volume_frame, text="0.0 L / 1000.0 L", 
                                    font=('Segoe UI', 11, 'bold'))
        self.volume_text = ttk.Label(volume_frame, text="0.0 L / 1000.0 L", 
                                    font=('Segoe UI', 11, 'bold'))
        self.volume_text.pack(side=tk.LEFT, padx=5)
        
        # Tasa de lluvia
        rain_rate_frame = ttk.Frame(indicator_frame)
        rain_rate_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(rain_rate_frame, text="Tasa de lluvia actual:", 
                 font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=5)
        
        self.rain_rate_text = ttk.Label(rain_rate_frame, text="0.0 mm/h", 
                                      font=('Segoe UI', 11, 'bold'))
        self.rain_rate_text.pack(side=tk.LEFT, padx=5)
        
        # Columna derecha: Indicador visual de nivel
        self.gauge_canvas = tk.Canvas(right_status, width=200, height=150, 
                                    background="#f5f5f5", highlightthickness=1, 
                                    highlightbackground="#cccccc")
        self.gauge_canvas.pack(padx=20, pady=10)
        
        # Dibujar medidor inicial
        self.draw_risk_gauge(0.0)
        
        # Panel inferior con pestañas para diferentes visualizaciones
        self.alerts_notebook = ttk.Notebook(main_frame)
        self.alerts_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Pestaña de historial de alertas
        self.alerts_history_tab = ttk.Frame(self.alerts_notebook)
        self.alerts_notebook.add(self.alerts_history_tab, text="Historial de Alertas")
        
        # Pestaña de gráficos de riesgo
        self.risk_chart_tab = ttk.Frame(self.alerts_notebook)
        self.alerts_notebook.add(self.risk_chart_tab, text="Gráficos de Riesgo")
        
        # Pestaña de configuración del sistema de alertas
        self.alerts_config_tab = ttk.Frame(self.alerts_notebook)
        self.alerts_notebook.add(self.alerts_config_tab, text="Configuración")
        
        # Configurar pestaña de historial de alertas
        self.setup_alerts_history_tab()
        
        # Configurar pestaña de gráficos de riesgo
        self.setup_risk_charts_tab()
        
        # Configurar pestaña de configuración
        self.setup_alerts_config_tab()
    
    def setup_alerts_history_tab(self):
        """Configura la pestaña de historial de alertas"""
        # Frame principal
        main_frame = ttk.Frame(self.alerts_history_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Lista de alertas con tabla
        alerts_frame = ttk.LabelFrame(main_frame, text="Historial de eventos de alerta", padding=10)
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Crear Treeview (tabla) para mostrar historial de alertas
        columns = ("fecha", "tipo", "descripcion", "volumen", "lluvia", "riesgo", "humedad")
        self.alerts_tree = ttk.Treeview(alerts_frame, columns=columns, show="headings", height=10)
        
        # Configurar columnas
        self.alerts_tree.heading("fecha", text="Fecha/Hora")
        self.alerts_tree.heading("tipo", text="Tipo Alerta")
        self.alerts_tree.heading("descripcion", text="Descripción")
        self.alerts_tree.heading("volumen", text="Volumen (L)")
        self.alerts_tree.heading("lluvia", text="Lluvia (mm/h)")
        self.alerts_tree.heading("riesgo", text="Riesgo (%)")
        self.alerts_tree.heading("humedad", text="Humedad (%)")
        
        # Configurar anchos de columna
        self.alerts_tree.column("fecha", width=150)
        self.alerts_tree.column("tipo", width=100)
        self.alerts_tree.column("descripcion", width=200)
        self.alerts_tree.column("volumen", width=80, anchor=tk.E)
        self.alerts_tree.column("lluvia", width=80, anchor=tk.E)
        self.alerts_tree.column("riesgo", width=80, anchor=tk.E)
        self.alerts_tree.column("humedad", width=80, anchor=tk.E)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL, command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscroll=scrollbar.set)
        
        # Colocar elementos
        self.alerts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configurar evento para mostrar detalles al seleccionar una alerta
        self.alerts_tree.bind('<<TreeviewSelect>>', self.show_alert_details)
        
        # Panel de detalles
        details_frame = ttk.LabelFrame(main_frame, text="Detalles de alerta seleccionada", padding=10)
        details_frame.pack(fill=tk.X, pady=5)
        
        self.alert_details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, height=5,
                                                         font=('Consolas', 10))
        self.alert_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Configurar filtros
        filter_frame = ttk.LabelFrame(main_frame, text="Filtros", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)
        
        # Tipos de alerta para filtrar
        ttk.Label(filter_frame, text="Tipo de alerta:").pack(side=tk.LEFT, padx=5)
        self.filter_type_var = tk.StringVar(value="TODOS")
        self.filter_type_combo = ttk.Combobox(filter_frame, textvariable=self.filter_type_var, 
                                            values=["TODOS", "VOLUMEN", "LLUVIA", "RIESGO", "MONITOR", "FIN"], 
                                            width=10)
        self.filter_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Rango de fechas
        ttk.Label(filter_frame, text="Desde:").pack(side=tk.LEFT, padx=(20, 5))
        self.filter_from_var = tk.StringVar(value=time.strftime("%Y/%m/%d"))
        ttk.Entry(filter_frame, textvariable=self.filter_from_var, width=12).pack(side=tk.LEFT)
        
        ttk.Label(filter_frame, text="Hasta:").pack(side=tk.LEFT, padx=(10, 5))
        self.filter_to_var = tk.StringVar(value=time.strftime("%Y/%m/%d"))
        ttk.Entry(filter_frame, textvariable=self.filter_to_var, width=12).pack(side=tk.LEFT)
        
        # Botón para aplicar filtros
        ttk.Button(filter_frame, text="Aplicar Filtros", 
                  command=self.apply_alert_filters).pack(side=tk.LEFT, padx=20)
    
    def setup_risk_charts_tab(self):
        """Configura la pestaña de gráficos de riesgo"""
        # Frame principal
        main_frame = ttk.Frame(self.risk_chart_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Área para gráficos
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Crear figura con varias subgráficas
        self.risk_fig, (self.risk_ax, self.volume_ax, self.rain_ax) = plt.subplots(
            3, 1, figsize=(10, 8), dpi=80, sharex=True)
        
        self.risk_canvas = FigureCanvasTkAgg(self.risk_fig, chart_frame)
        self.risk_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurar gráficos
        self.risk_ax.set_title('Nivel de Riesgo (%)')
        self.risk_ax.grid(True)
        
        self.volume_ax.set_title('Volumen de Agua (L)')
        self.volume_ax.grid(True)
        
        self.rain_ax.set_title('Tasa de Lluvia (mm/h)')
        self.rain_ax.grid(True)
        self.rain_ax.set_xlabel('Fecha/Hora')
        
        # Ajustar diseño
        self.risk_fig.tight_layout()
        
        # Panel de control
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Rango de tiempo para mostrar
        ttk.Label(control_frame, text="Mostrar:").pack(side=tk.LEFT, padx=5)
        self.chart_range_var = tk.StringVar(value="24h")
        ttk.Combobox(control_frame, textvariable=self.chart_range_var, 
                    values=["1h", "6h", "12h", "24h", "48h", "7d", "Todo"], 
                    width=5).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Actualizar Gráficos", 
                  command=self.update_risk_charts).pack(side=tk.LEFT, padx=20)
    
    def setup_alerts_config_tab(self):
        """Configura la pestaña de configuración del sistema de alertas"""
        # Frame principal
        main_frame = ttk.Frame(self.alerts_config_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Parámetros de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Parámetros del Sistema de Alertas", padding=10)
        config_frame.pack(fill=tk.X, pady=5, expand=False)
        
        # Área de recolección
        area_frame = ttk.Frame(config_frame)
        area_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(area_frame, text="Área de recolección:").pack(side=tk.LEFT, padx=5)
        self.config_area_var = tk.StringVar(value="18.0")
        ttk.Entry(area_frame, textvariable=self.config_area_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(area_frame, text="m²").pack(side=tk.LEFT, padx=2)
        
        # Capacidad del sistema
        capacity_frame = ttk.Frame(config_frame)
        capacity_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(capacity_frame, text="Capacidad del sistema:").pack(side=tk.LEFT, padx=5)
        self.config_capacity_var = tk.StringVar(value="1000.0")
        ttk.Entry(capacity_frame, textvariable=self.config_capacity_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(capacity_frame, text="litros").pack(side=tk.LEFT, padx=2)
        
        # Botones de acción
        buttons_frame = ttk.Frame(config_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Leer Configuración Actual", 
                  command=self.read_current_config,
                  style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="Actualizar Configuración", 
                  command=self.update_alert_config,
                  style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        # Simulador de alertas (para pruebas)
        simulator_frame = ttk.LabelFrame(main_frame, text="Simulador de Alertas (Pruebas)", padding=10)
        simulator_frame.pack(fill=tk.X, pady=10, expand=False)
        
        # Valores para la simulación
        sim_row1 = ttk.Frame(simulator_frame)
        sim_row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(sim_row1, text="Volumen (L):").pack(side=tk.LEFT, padx=5)
        self.sim_volume_var = tk.StringVar(value="600.0")
        ttk.Entry(sim_row1, textvariable=self.sim_volume_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sim_row1, text="Lluvia (mm/h):").pack(side=tk.LEFT, padx=(20, 5))
        self.sim_rain_var = tk.StringVar(value="15.0")
        ttk.Entry(sim_row1, textvariable=self.sim_rain_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sim_row1, text="Humedad (%):").pack(side=tk.LEFT, padx=(20, 5))
        self.sim_humidity_var = tk.StringVar(value="85.0")
        ttk.Entry(sim_row1, textvariable=self.sim_humidity_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Botones de simulación
        sim_row2 = ttk.Frame(simulator_frame)
        sim_row2.pack(fill=tk.X, pady=5)
        
        ttk.Button(sim_row2, text="Simular Alerta", 
                  command=self.simulate_alert).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(sim_row2, text="Detener Simulación", 
                  command=self.stop_alert_simulation).pack(side=tk.LEFT, padx=5)
        
        # Documentación del sistema de alertas
        doc_frame = ttk.LabelFrame(main_frame, text="Documentación del Sistema de Alertas", padding=10)
        doc_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        doc_text = scrolledtext.ScrolledText(doc_frame, wrap=tk.WORD, height=8, 
                                           font=('Segoe UI', 10))
        doc_text.pack(fill=tk.BOTH, expand=True)
        
        # Texto de documentación
        doc_text.insert(tk.END, """El sistema de alertas utiliza lógica difusa para evaluar el riesgo de desbordamiento basándose en tres variables principales:

1. Volumen actual: El volumen de agua acumulada en el sistema de recolección.
2. Tasa de lluvia: La intensidad de precipitación actual en mm/h.
3. Humedad ambiental: El nivel de humedad del aire en porcentaje.

Tipos de alertas:
- VOLUMEN: Se activa cuando el volumen supera el 75% de la capacidad del sistema.
- LLUVIA: Se activa cuando la intensidad de lluvia supera los 25 mm/h.
- RIESGO: Se activa cuando el cálculo de riesgo difuso supera el 65% y la humedad es alta.

El sistema monitorea continuamente estas variables y utiliza un algoritmo de lógica difusa para calcular el nivel de riesgo global en una escala de 0 a 100%. Las alertas se registran en un archivo dedicado para facilitar el análisis posterior.""")
        doc_text.config(state=tk.DISABLED)  # Hacer el texto de solo lectura
    
    def setup_water_calculation_tab(self):
        """Configura la pestaña de cálculo de agua recolectada"""
        main_frame = ttk.Frame(self.water_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título y descripción
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(title_frame, text="Cálculo de Agua Recolectada - Estación Lestoma", 
                font=('Segoe UI', 14, 'bold')).pack(anchor=tk.CENTER)
        
        ttk.Label(title_frame, text="Calcule el volumen de agua recolectada para un día específico",
                font=('Segoe UI', 11)).pack(anchor=tk.CENTER, pady=2)
        
        # Dividir en dos paneles: izquierdo para parámetros, derecho para gráfico
        panels_frame = ttk.Frame(main_frame)
        panels_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Panel izquierdo - parámetros y cálculo
        left_panel = ttk.Frame(panels_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_panel.pack_propagate(False)  # Esto evita que el frame se redimensione
        
        # Marco para parámetros
        params_frame = ttk.LabelFrame(left_panel, text="Parámetros de Cálculo", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Área de recolección
        area_frame = ttk.Frame(params_frame)
        area_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(area_frame, text="Área de recolección:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        self.area_var = tk.StringVar(value="18")
        ttk.Entry(area_frame, textvariable=self.area_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(area_frame, text="m²", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        
        # Selección de fecha
        date_frame = ttk.Frame(params_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Fecha (YYYY/MM/DD):", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        self.calc_date_var = tk.StringVar(value=time.strftime("%Y/%m/%d"))
        date_entry = ttk.Entry(date_frame, textvariable=self.calc_date_var, width=15)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(date_frame, text="Hoy", 
                command=lambda: self.calc_date_var.set(time.strftime("%Y/%m/%d"))).pack(side=tk.LEFT, padx=5)
        
        # Botones de acción
        action_frame = ttk.Frame(params_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Descargar Datos del Día", 
                command=self.download_day_data, style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="Exportar Resultados", 
                command=self.export_water_calculation, style="Command.TButton").pack(side=tk.LEFT, padx=5)
        
        # Botón de cálculo prominente
        calc_button_frame = ttk.Frame(params_frame)
        calc_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(calc_button_frame, text="CALCULAR AGUA RECOLECTADA", 
                command=self.calculate_water_volume, 
                style='Big.TButton').pack(anchor=tk.CENTER, pady=5)
        
        # Resultados destacados
        highlight_frame = ttk.LabelFrame(left_panel, text="Resultados del Cálculo", padding=10)
        highlight_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Precipitación
        precip_frame = ttk.Frame(highlight_frame)
        precip_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(precip_frame, text="Precipitación Total:", 
                font=('Segoe UI', 12, 'bold')).pack(anchor=tk.CENTER)
        self.result_precip = ttk.Label(precip_frame, text="--.- mm", 
                                    style='Reading.TLabel')
        self.result_precip.pack(anchor=tk.CENTER, pady=2)
        
        # Volumen
        volume_frame = ttk.Frame(highlight_frame)
        volume_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(volume_frame, text="Volumen de Agua Recolectada:", 
                font=('Segoe UI', 12, 'bold')).pack(anchor=tk.CENTER)
        self.result_volume = ttk.Label(volume_frame, text="--.- L", 
                                    style='Result.TLabel')
        self.result_volume.pack(anchor=tk.CENTER, pady=2)
        
        # Visualización gráfica de agua recolectada
        water_vis_frame = ttk.Frame(highlight_frame)
        water_vis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.water_canvas = tk.Canvas(water_vis_frame, height=150, background="#f5f5f5", 
                                    highlightthickness=1, highlightbackground="#cccccc")
        self.water_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel derecho - gráfico de precipitación y tabla (asegurar que tome más espacio)
        right_panel = ttk.Frame(panels_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Marco para gráfico de precipitación diaria
        rain_chart_container = ttk.LabelFrame(right_panel, text="Precipitación Diaria", padding=5)
        rain_chart_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Crear contenedor para el gráfico
        self.rain_chart_frame = ttk.Frame(rain_chart_container)
        self.rain_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Inicializar figura para gráfico de precipitación
        self.rain_day_fig, self.rain_day_ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.rain_day_canvas = FigureCanvasTkAgg(self.rain_day_fig, self.rain_chart_frame)
        self.rain_day_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.rain_day_ax.set_title('Precipitación del Día (mm)')
        self.rain_day_ax.grid(True)
        self.rain_day_fig.tight_layout()
        
        # Tabla de resultados
        self.result_frame = ttk.LabelFrame(right_panel, text="Detalles de Registros", padding=5)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(self.result_frame, wrap=tk.WORD, height=10, 
                                                font=('Consolas', 9))
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def draw_risk_gauge(self, risk_value):
        """Dibuja un medidor de riesgo en el canvas"""
        # Limpiar canvas
        self.gauge_canvas.delete("all")
        
        # Configuración
        canvas_width = self.gauge_canvas.winfo_width()
        canvas_height = self.gauge_canvas.winfo_height()
        
        if canvas_width < 50 or canvas_height < 50:  # Si el canvas aún no tiene dimensiones adecuadas
            canvas_width = 200
            canvas_height = 150
        
        # Definir parámetros
        center_x = canvas_width / 2
        center_y = canvas_height - 20
        radius = min(canvas_width, canvas_height * 2) / 2 - 10
        
        # Convertir valor de riesgo a ángulo (0-100% → 180°-0°)
        angle = 180 - (risk_value * 180 / 100)
        
        # Dibujar arco de fondo
        self.gauge_canvas.create_arc(center_x - radius, center_y - radius, 
                                   center_x + radius, center_y + radius,
                                   start=0, extent=180, style=tk.ARC, width=10,
                                   outline="#d0d0d0")
        
        # Determinar color según nivel de riesgo
        if risk_value < 40:
            color = "#2ecc71"  # Verde
        elif risk_value < 70:
            color = "#f39c12"  # Naranja
        else:
            color = "#e74c3c"  # Rojo
        
        # Dibujar arco de riesgo actual
        self.gauge_canvas.create_arc(center_x - radius, center_y - radius, 
                                   center_x + radius, center_y + radius,
                                   start=180, extent=-180+angle, style=tk.ARC, width=10,
                                   outline=color)
        
        # Dibujar aguja indicadora
        angle_rad = math.radians(angle)
        # Calcular punto externo del indicador
        x_outer = center_x + (radius - 5) * math.cos(angle_rad)
        y_outer = center_y - (radius - 5) * math.sin(angle_rad)
        # Calcular puntos base del indicador
        base_width = 8
        base_x1 = center_x - base_width
        base_x2 = center_x + base_width
        
        # Dibujar indicador como triángulo
        self.gauge_canvas.create_polygon(
            base_x1, center_y, base_x2, center_y, x_outer, y_outer,
            fill=color, outline=color, width=2)
        
        # Dibujar círculo central
        self.gauge_canvas.create_oval(center_x-10, center_y-10, center_x+10, center_y+10,
                                    fill="#ffffff", outline="#333333")
        
        # Dibujar marcas de escala
        for i in range(0, 101, 20):
            scale_angle = 180 - (i * 180 / 100)
            scale_angle_rad = math.radians(scale_angle)
            
            x_inner = center_x + (radius - 15) * math.cos(scale_angle_rad)
            y_inner = center_y - (radius - 15) * math.sin(scale_angle_rad)
            
            x_outer = center_x + (radius + 5) * math.cos(scale_angle_rad)
            y_outer = center_y - (radius + 5) * math.sin(scale_angle_rad)
            
            self.gauge_canvas.create_line(x_inner, y_inner, x_outer, y_outer, fill="#333333", width=2)
            
            # Etiqueta de valor
            label_x = center_x + (radius + 20) * math.cos(scale_angle_rad)
            label_y = center_y - (radius + 20) * math.sin(scale_angle_rad)
            
            self.gauge_canvas.create_text(label_x, label_y, text=f"{i}%", 
                                        font=("Segoe UI", 8), fill="#333333")
        
        # Dibujar valor de riesgo actual
        self.gauge_canvas.create_text(center_x, center_y - 30, 
                                     text=f"Riesgo: {risk_value:.1f}%",
                                     font=("Segoe UI", 12, "bold"), fill=color)
    
    def init_charts(self):
        """Inicializa los gráficos vacíos"""
        # Configurar estilo de gráficos
        plt.style.use('ggplot')
        
        # Temperatura y humedad
        self.temp_hum_fig, (self.temp_ax, self.hum_ax) = plt.subplots(2, 1, figsize=(9, 7), dpi=80)
        self.temp_hum_canvas = FigureCanvasTkAgg(self.temp_hum_fig, self.temp_hum_tab)
        self.temp_hum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.temp_ax.set_title('Temperatura (°C)')
        self.temp_ax.grid(True)
        self.hum_ax.set_title('Humedad Relativa (%)')
        self.hum_ax.grid(True)
        
        # Precipitación
        self.rain_fig, self.rain_ax = plt.subplots(figsize=(9, 6), dpi=80)
        self.rain_canvas = FigureCanvasTkAgg(self.rain_fig, self.rain_tab)
        self.rain_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.rain_ax.set_title('Precipitación (mm)')
        self.rain_ax.grid(True)
        
        # Viento
        self.wind_fig, (self.wind_speed_ax, self.wind_dir_ax) = plt.subplots(2, 1, figsize=(9, 7), dpi=80)
        self.wind_canvas = FigureCanvasTkAgg(self.wind_fig, self.wind_tab)
        self.wind_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.wind_speed_ax.set_title('Velocidad del Viento (km/h)')
        self.wind_speed_ax.grid(True)
        self.wind_dir_ax.set_title('Dirección del Viento')
        self.wind_dir_ax.grid(True)
        
        # Radiación/Luminosidad
        self.rad_fig, (self.rad_ax, self.lux_ax) = plt.subplots(2, 1, figsize=(9, 7), dpi=80)
        self.rad_canvas = FigureCanvasTkAgg(self.rad_fig, self.rad_tab)
        self.rad_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.rad_ax.set_title('Radiación Solar (W/m²)')
        self.rad_ax.grid(True)
        self.lux_ax.set_title('Luminosidad (lux)')
        self.lux_ax.grid(True)
        
        # Ajustar espaciado
        self.temp_hum_fig.tight_layout()
        self.rain_fig.tight_layout()
        self.wind_fig.tight_layout()
        self.rad_fig.tight_layout()
    
    def toggle_auto_update(self):
        """Activa o desactiva la actualización automática"""
        if self.auto_update_var.get():
            # Activar actualización automática
            self.schedule_auto_update()
            self.add_log("Actualización automática activada")
        else:
            # Desactivar actualización automática
            if self.auto_update_job:
                self.root.after_cancel(self.auto_update_job)
                self.auto_update_job = None
            self.add_log("Actualización automática desactivada")
    
    def schedule_auto_update(self):
        """Programa la próxima actualización automática"""
        if self.auto_update_var.get() and self.connected:
            # Obtener intervalo en segundos
            try:
                interval = int(self.update_interval.get()) * 1000  # convertir a milisegundos
                self.send_predefined_command("DATOS")
                self.auto_update_job = self.root.after(interval, self.schedule_auto_update)
            except:
                self.add_log("Error: Intervalo de actualización inválido")
    
    def apply_alert_filters(self):
        """Aplica los filtros seleccionados al historial de alertas"""
        try:
            # Limpiar tabla actual
            for item in self.alerts_tree.get_children():
                self.alerts_tree.delete(item)
            
            if not self.alertas_data['timestamp']:
                self.add_log("No hay datos de alertas para filtrar")
                return
            
            # Obtener valores de filtro
            tipo_filtro = self.filter_type_var.get()
            fecha_desde = self.filter_from_var.get()
            fecha_hasta = self.filter_to_var.get()
            
            # Convertir fechas a objetos datetime para comparación
            try:
                desde_dt = datetime.strptime(fecha_desde, '%Y/%m/%d') if fecha_desde else None
                hasta_dt = datetime.strptime(fecha_hasta, '%Y/%m/%d') if fecha_hasta else None
                
                # Ajustar hasta_dt para incluir todo el día
                if hasta_dt:
                    hasta_dt = hasta_dt.replace(hour=23, minute=59, second=59)
            except ValueError:
                self.add_log("Error: Formato de fecha incorrecto. Use YYYY/MM/DD")
                return
            
            # Aplicar filtros
            for i in range(len(self.alertas_data['timestamp'])):
                # Verificar tipo de alerta
                if tipo_filtro != "TODOS" and self.alertas_data['tipo_alerta'][i] != tipo_filtro:
                    continue
                
                # Verificar rango de fechas
                if self.alertas_data['timestamp'][i]:
                    try:
                        alerta_dt = datetime.strptime(self.alertas_data['timestamp'][i], '%Y-%m-%d %H:%M:%S')
                        if desde_dt and alerta_dt < desde_dt:
                            continue
                        if hasta_dt and alerta_dt > hasta_dt:
                            continue
                    except:
                        # Si hay error en el formato de fecha, seguimos adelante
                        pass
                
                # Si pasa todos los filtros, añadir a la tabla
                self.alerts_tree.insert('', tk.END, values=(
                    self.alertas_data['timestamp'][i],
                    self.alertas_data['tipo_alerta'][i],
                    self.alertas_data['descripcion'][i],
                    self.alertas_data['volumen_l'][i],
                    self.alertas_data['lluvia_mm_h'][i],
                    self.alertas_data['riesgo'][i],
                    self.alertas_data['humedad'][i]
                ))
            
            # Actualizar información
            num_items = len(self.alerts_tree.get_children())
            self.add_log(f"Filtros aplicados: se muestran {num_items} alertas")
            
        except Exception as e:
            self.add_log(f"Error al aplicar filtros: {str(e)}")
    
    def show_alert_details(self, event):
        """Muestra detalles de la alerta seleccionada"""
        selected_items = self.alerts_tree.selection()
        if not selected_items:
            return
        
        item = selected_items[0]
        values = self.alerts_tree.item(item, 'values')
        
        # Limpiar detalles anteriores
        self.alert_details_text.delete(1.0, tk.END)
        
        # Mostrar detalles formateados
        self.alert_details_text.insert(tk.END, "DETALLES DE ALERTA\n")
        self.alert_details_text.insert(tk.END, "=====================================\n")
        self.alert_details_text.insert(tk.END, f"Fecha/Hora: {values[0]}\n")
        self.alert_details_text.insert(tk.END, f"Tipo: {values[1]}\n")
        self.alert_details_text.insert(tk.END, f"Descripción: {values[2]}\n\n")
        
        self.alert_details_text.insert(tk.END, "PARÁMETROS DEL SISTEMA\n")
        self.alert_details_text.insert(tk.END, "------------------------------------\n")
        self.alert_details_text.insert(tk.END, f"Volumen de agua: {values[3]} L\n")
        self.alert_details_text.insert(tk.END, f"Tasa de lluvia: {values[4]} mm/h\n")
        self.alert_details_text.insert(tk.END, f"Nivel de riesgo: {values[5]}%\n")
        self.alert_details_text.insert(tk.END, f"Humedad relativa: {values[6]}%\n\n")
        
        # Añadir análisis según el tipo de alerta
        self.alert_details_text.insert(tk.END, "ANÁLISIS\n")
        self.alert_details_text.insert(tk.END, "------------------------------------\n")
        
        tipo_alerta = values[1]
        if tipo_alerta == "VOLUMEN":
            self.alert_details_text.insert(tk.END, "Alerta por VOLUMEN CRÍTICO: El sistema está cerca de su capacidad máxima.\n")
            self.alert_details_text.insert(tk.END, "Se recomienda evacuar agua del sistema o ampliar la capacidad de almacenamiento.\n")
        elif tipo_alerta == "LLUVIA":
            self.alert_details_text.insert(tk.END, "Alerta por LLUVIA INTENSA: La tasa de precipitación es elevada.\n")
            self.alert_details_text.insert(tk.END, "Monitorear acumulación de agua y posible saturación del sistema.\n")
        elif tipo_alerta == "RIESGO":
            self.alert_details_text.insert(tk.END, "Alerta por RIESGO COMBINADO: Múltiples factores indican posible desbordamiento.\n")
            self.alert_details_text.insert(tk.END, "Verificar volumen, lluvia y condiciones climáticas para tomar medidas preventivas.\n")
        elif tipo_alerta == "MONITOR":
            self.alert_details_text.insert(tk.END, "Monitoreo de alerta activa: El sistema continúa en estado de alerta.\n")
        elif tipo_alerta == "FIN":
            self.alert_details_text.insert(tk.END, "Fin de alerta: Los parámetros han vuelto a niveles seguros.\n")
    
    def update_risk_charts(self):
        """Actualiza los gráficos de riesgo con datos filtrados por tiempo"""
        if not self.alertas_data['timestamp']:
            self.add_log("No hay datos de alertas para graficar")
            return
        
        try:
            # Convertir timestamps a objetos datetime
            dates = []
            for d in self.alertas_data['timestamp']:
                try:
                    dates.append(datetime.strptime(d, "%Y-%m-%d %H:%M:%S"))
                except:
                    dates.append(None)
            
            # Filtrar datos válidos
            valid_indices = [i for i, d in enumerate(dates) if d is not None]
            if not valid_indices:
                self.add_log("No hay fechas válidas para graficar")
                return
            
            valid_dates = [dates[i] for i in valid_indices]
            valid_risk = [float(self.alertas_data['riesgo'][i]) if self.alertas_data['riesgo'][i] else 0 
                          for i in valid_indices]
            valid_volume = [float(self.alertas_data['volumen_l'][i]) if self.alertas_data['volumen_l'][i] else 0 
                           for i in valid_indices]
            valid_rain = [float(self.alertas_data['lluvia_mm_h'][i]) if self.alertas_data['lluvia_mm_h'][i] else 0 
                         for i in valid_indices]
            
            # Aplicar filtro de tiempo
            time_range = self.chart_range_var.get()
            now = datetime.now()
            
            if time_range != "Todo":
                cutoff_date = None
                if time_range == "1h":
                    cutoff_date = now - pd.Timedelta(hours=1)
                elif time_range == "6h":
                    cutoff_date = now - pd.Timedelta(hours=6)
                elif time_range == "12h":
                    cutoff_date = now - pd.Timedelta(hours=12)
                elif time_range == "24h":
                    cutoff_date = now - pd.Timedelta(hours=24)
                elif time_range == "48h":
                    cutoff_date = now - pd.Timedelta(hours=48)
                elif time_range == "7d":
                    cutoff_date = now - pd.Timedelta(days=7)
                
                if cutoff_date:
                    filtered_indices = [i for i, date in enumerate(valid_dates) if date >= cutoff_date]
                    valid_dates = [valid_dates[i] for i in filtered_indices]
                    valid_risk = [valid_risk[i] for i in filtered_indices]
                    valid_volume = [valid_volume[i] for i in filtered_indices]
                    valid_rain = [valid_rain[i] for i in filtered_indices]
            
            # Limpiar gráficos
            self.risk_ax.clear()
            self.volume_ax.clear()
            self.rain_ax.clear()
            
            # Graficar datos filtrados
            if valid_dates:
                # Riesgo
                self.risk_ax.plot(valid_dates, valid_risk, 'r-', marker='o', markersize=3)
                self.risk_ax.fill_between(valid_dates, 0, valid_risk, alpha=0.2, color='red')
                self.risk_ax.set_title('Nivel de Riesgo (%)')
                self.risk_ax.set_ylabel('%')
                self.risk_ax.grid(True, linestyle='--', alpha=0.7)
                self.risk_ax.set_ylim(0, 100)
                
                # Añadir línea de umbral de riesgo
                self.risk_ax.axhline(y=65, color='r', linestyle='--', alpha=0.8)
                self.risk_ax.text(valid_dates[0], 67, 'Umbral de riesgo (65%)', 
                                 fontsize=8, color='red')
                
                # Volumen
                self.volume_ax.plot(valid_dates, valid_volume, 'b-', marker='o', markersize=3)
                self.volume_ax.fill_between(valid_dates, 0, valid_volume, alpha=0.2, color='blue')
                self.volume_ax.set_title('Volumen de Agua (L)')
                self.volume_ax.set_ylabel('Litros')
                self.volume_ax.grid(True, linestyle='--', alpha=0.7)
                
                # Añadir línea de capacidad
                capacidad = float(self.config_capacity_var.get())
                self.volume_ax.axhline(y=capacidad, color='b', linestyle='--', alpha=0.8)
                self.volume_ax.text(valid_dates[0], capacidad*1.02, f'Capacidad ({capacidad} L)', 
                                  fontsize=8, color='blue')
                
                # Añadir línea de umbral crítico (75%)
                umbral = capacidad * 0.75
                self.volume_ax.axhline(y=umbral, color='orange', linestyle='--', alpha=0.8)
                self.volume_ax.text(valid_dates[0], umbral*1.02, f'Umbral crítico ({umbral} L)', 
                                  fontsize=8, color='orange')
                
                # Lluvia
                self.rain_ax.plot(valid_dates, valid_rain, 'g-', marker='o', markersize=3)
                self.rain_ax.fill_between(valid_dates, 0, valid_rain, alpha=0.2, color='green')
                self.rain_ax.set_title('Tasa de Lluvia (mm/h)')
                self.rain_ax.set_ylabel('mm/h')
                self.rain_ax.set_xlabel('Fecha/Hora')
                self.rain_ax.grid(True, linestyle='--', alpha=0.7)
                
                # Añadir línea de lluvia intensa
                self.rain_ax.axhline(y=25, color='g', linestyle='--', alpha=0.8)
                self.rain_ax.text(valid_dates[0], 26, 'Lluvia intensa (25 mm/h)', 
                                fontsize=8, color='green')
                
                # Configurar formato de fecha para todos los ejes
                date_fmt = mdates.DateFormatter('%d/%m %H:%M')
                for ax in [self.risk_ax, self.volume_ax, self.rain_ax]:
                    ax.xaxis.set_major_formatter(date_fmt)
                    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                
                # Ajustar diseño
                self.risk_fig.tight_layout()
                self.risk_canvas.draw()
                
                self.add_log(f"Gráficos de riesgo actualizados. Mostrando {len(valid_dates)} registros en rango '{time_range}'")
            else:
                self.add_log("No hay datos suficientes para generar gráficos en el rango seleccionado")
        
        except Exception as e:
            self.add_log(f"Error al actualizar gráficos de riesgo: {str(e)}")
            import traceback
            self.add_log(traceback.format_exc())
    
    def read_current_config(self):
        """Lee la configuración actual del sistema desde la estación"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        self.add_log("Solicitando configuración actual...")
        self.send_predefined_command("INFO")
    
    def update_alert_config(self):
        """Actualiza la configuración del sistema de alertas en la estación"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        try:
            # Validar valores
            area = float(self.config_area_var.get())
            capacity = float(self.config_capacity_var.get())
            
            if area <= 0 or capacity <= 0:
                messagebox.showerror("Error", "Los valores deben ser mayores que cero")
                return
            
            # Enviar comandos para actualizar
            self.add_log(f"Actualizando configuración: Área={area}m², Capacidad={capacity}L")
            self.send_predefined_command(f"SET AREA {area}")
            
            # Breve pausa para no enviar comandos demasiado seguidos
            self.root.after(1000, lambda: self.send_predefined_command(f"SET CAPACIDAD {capacity}"))
            
            # Verificar cambios después de unos segundos
            self.root.after(3000, self.read_current_config)
            
        except ValueError:
            messagebox.showerror("Error", "Valores inválidos. Ingrese números válidos.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al actualizar configuración: {str(e)}")
    
    def simulate_alert(self):
        """Simula una alerta para pruebas (solo actualiza la interfaz)"""
        try:
            volume = float(self.sim_volume_var.get())
            rain = float(self.sim_rain_var.get())
            humidity = float(self.sim_humidity_var.get())
            
            # Calcular riesgo simulado (simplificado para la interfaz)
            risk = min(100, max(0, (volume/float(self.config_capacity_var.get()))*50 + 
                              (rain/25)*30 + (humidity/100)*20))
            
            self.add_log(f"Simulando alerta: Vol={volume}L, Lluvia={rain}mm/h, Humedad={humidity}%")
            
            # Actualizar indicadores
            self.volume_text.config(text=f"{volume:.1f} L / {self.config_capacity_var.get()} L")
            self.rain_rate_text.config(text=f"{rain:.1f} mm/h")
            self.risk_level_text.config(text=f"{risk:.1f}%")
            
            # Determinar tipo de alerta
            alert_type = []
            if volume >= float(self.config_capacity_var.get()) * 0.75:
                alert_type.append("VOLUMEN")
            if rain >= 25:
                alert_type.append("LLUVIA")
            if risk >= 65 and humidity >= 80:
                alert_type.append("RIESGO")
            
            # Activar indicadores de alerta si corresponde
            if alert_type:
                self.alert_active = True
                tipo = "_".join(alert_type)
                self.alert_state_canvas.configure(bg="#e74c3c")
                self.alert_state_text.configure(text=f"ALERTA ACTIVA: {tipo}")
                self.alerts_title.configure(text=f"Sistema de Alertas - ¡ALERTA ACTIVA! {tipo}")
                
                # También actualizar indicador del dashboard
                self.alert_indicator.configure(bg="#e74c3c")
                self.alert_status_label.configure(text=f"¡ALERTA! {tipo}")
                
                # Mostrar diálogo de alerta
                if not hasattr(self, 'alert_window_active') or not self.alert_window_active:
                    self.show_alert_notification(f"Alerta tipo: {tipo}", 
                                              f"Volumen: {volume:.1f}L, Lluvia: {rain:.1f}mm/h, Riesgo: {risk:.1f}%")
            else:
                self.alert_active = False
                self.alert_state_canvas.configure(bg="#2ecc71")
                self.alert_state_text.configure(text="Normal - sin alertas")
                self.alerts_title.configure(text="Sistema de Alertas - Monitoreo")
                
                # También actualizar indicador del dashboard
                self.alert_indicator.configure(bg="#2ecc71")
                self.alert_status_label.configure(text="Normal")
            
            # Actualizar medidor de riesgo
            self.draw_risk_gauge(risk)
            
            # Añadir entrada simulada a la tabla de alertas
            if alert_type:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.alertas_data['timestamp'].append(timestamp)
                self.alertas_data['tipo_alerta'].append("_".join(alert_type))
                self.alertas_data['descripcion'].append("Alerta simulada para pruebas")
                self.alertas_data['volumen_l'].append(str(volume))
                self.alertas_data['lluvia_mm_h'].append(str(rain))
                self.alertas_data['riesgo'].append(str(risk))
                self.alertas_data['humedad'].append(str(humidity))
                
                # Actualizar vista de alertas
                self.apply_alert_filters()
                self.update_risk_charts()
            
        except ValueError:
            messagebox.showerror("Error", "Valores inválidos para simulación")
        except Exception as e:
            messagebox.showerror("Error", f"Error en simulación: {str(e)}")
    
    def stop_alert_simulation(self):
        """Detiene la simulación de alerta"""
        self.alert_active = False
        self.alert_state_canvas.configure(bg="#cccccc")
        self.alert_state_text.configure(text="Sin alertas activas")
        self.alerts_title.configure(text="Sistema de Alertas - Monitoreo")
        
        # Reiniciar indicadores
        self.volume_text.config(text="0.0 L / 1000.0 L")
        self.rain_rate_text.config(text="0.0 mm/h")
        self.risk_level_text.config(text="0.0%")
        
        # Actualizar medidor de riesgo
        self.draw_risk_gauge(0)
        
        # Actualizar indicador del dashboard
        self.alert_indicator.configure(bg="#cccccc")
        self.alert_status_label.configure(text="Sin alertas")
        
        self.add_log("Simulación de alerta detenida")
        
        # Añadir entrada de fin de alerta si había una activa
        if hasattr(self, 'alert_window_active') and self.alert_window_active:
            if hasattr(self, 'alert_window'):
                try:
                    self.alert_window.destroy()
                except:
                    pass
            self.alert_window_active = False
    
    def check_for_alerts(self):
        """Verifica periódicamente si hay alertas nuevas"""
        # Verificar si hay alertas activas
        if self.connected and self.alertas_data['timestamp']:
            # Buscar alertas recientes (últimos 5 minutos)
            now = datetime.now()
            cutoff = now - pd.Timedelta(minutes=5)
            
            try:
                for i in range(len(self.alertas_data['timestamp'])-1, -1, -1):
                    # Solo verificar las últimas 10 alertas como máximo
                    if i < len(self.alertas_data['timestamp']) - 10:
                        break
                    
                    try:
                        alert_time = datetime.strptime(self.alertas_data['timestamp'][i], "%Y-%m-%d %H:%M:%S")
                        
                        # Si es una alerta reciente y no es de tipo "FIN" o "MONITOR"
                        if (alert_time > cutoff and 
                            self.alertas_data['tipo_alerta'][i] not in ["FIN", "MONITOR"]):
                            
                            # Activar indicadores si no están ya activos
                            if not self.alert_active:
                                self.alert_active = True
                                self.alert_state_canvas.configure(bg="#e74c3c")
                                self.alert_state_text.configure(text=f"ALERTA ACTIVA: {self.alertas_data['tipo_alerta'][i]}")
                                self.alerts_title.configure(text=f"Sistema de Alertas - ¡ALERTA ACTIVA!")
                                
                                # También actualizar indicador del dashboard
                                self.alert_indicator.configure(bg="#e74c3c")
                                self.alert_status_label.configure(text=f"¡ALERTA! {self.alertas_data['tipo_alerta'][i]}")
                                
                                # Mostrar diálogo de alerta
                                self.show_alert_notification(
                                    f"Alerta tipo: {self.alertas_data['tipo_alerta'][i]}", 
                                    f"Descripción: {self.alertas_data['descripcion'][i]}\n" +
                                    f"Volumen: {self.alertas_data['volumen_l'][i]}L\n" +
                                    f"Lluvia: {self.alertas_data['lluvia_mm_h'][i]}mm/h\n" +
                                    f"Riesgo: {self.alertas_data['riesgo'][i]}%"
                                )
                                
                                # Actualizar medidor de riesgo
                                if self.alertas_data['riesgo'][i]:
                                    try:
                                        self.draw_risk_gauge(float(self.alertas_data['riesgo'][i]))
                                    except:
                                        pass
                                
                                # Actualizar volumen y lluvia
                                try:
                                    self.volume_text.config(
                                        text=f"{float(self.alertas_data['volumen_l'][i]):.1f} L / {self.config_capacity_var.get()} L")
                                    self.rain_rate_text.config(
                                        text=f"{float(self.alertas_data['lluvia_mm_h'][i]):.1f} mm/h")
                                    self.risk_level_text.config(
                                        text=f"{float(self.alertas_data['riesgo'][i]):.1f}%")
                                except:
                                    pass
                                
                                # Detener búsqueda
                                break
                    except:
                        # Error en formato de fecha, continuar con el siguiente
                        continue
            except Exception as e:
                print(f"Error en check_for_alerts: {e}")
        
        # Programar próxima verificación
        self.root.after(self.alert_check_interval, self.check_for_alerts)
    
    def show_alert_notification(self, title, message):
        """Muestra una ventana de notificación para alertas importantes"""
        try:
            # Verificar si ya existe una ventana de alerta
            if hasattr(self, 'alert_window_active') and self.alert_window_active:
                # Actualizar la ventana existente
                if hasattr(self, 'alert_window') and self.alert_window.winfo_exists():
                    self.alert_message.config(text=message)
                    return
            
            # Crear nueva ventana de alerta
            self.alert_window = tk.Toplevel(self.root)
            self.alert_window.title("¡ALERTA DEL SISTEMA!")
            self.alert_window.geometry("400x300")
            self.alert_window.configure(bg="#ffcccc")
            self.alert_window.attributes('-topmost', True)
            self.alert_window_active = True
            
            # Centrar en pantalla
            self.alert_window.update_idletasks()
            width = self.alert_window.winfo_width()
            height = self.alert_window.winfo_height()
            x = (self.alert_window.winfo_screenwidth() // 2) - (width // 2)
            y = (self.alert_window.winfo_screenheight() // 2) - (height // 2)
            self.alert_window.geometry(f'{width}x{height}+{x}+{y}')
            
            # Contenido
            alert_frame = tk.Frame(self.alert_window, bg="#ffcccc", padx=20, pady=20)
            alert_frame.pack(fill=tk.BOTH, expand=True)
            
            # Icono de advertencia (emoji)
            warning_label = tk.Label(alert_frame, text="⚠️", font=("Arial", 48), bg="#ffcccc")
            warning_label.pack(pady=10)
            
            # Título de alerta
            alert_title = tk.Label(alert_frame, text=title, font=("Arial", 14, "bold"), 
                                 bg="#ffcccc", fg="#cc0000")
            alert_title.pack(pady=5)
            
            # Mensaje
            self.alert_message = tk.Label(alert_frame, text=message, font=("Arial", 11), 
                                      bg="#ffcccc", justify=tk.LEFT, wraplength=350)
            self.alert_message.pack(pady=10)
            
            # Botón para reconocer la alerta
            acknowledge_btn = tk.Button(alert_frame, text="Reconocer Alerta", 
                                      font=("Arial", 12, "bold"),
                                      command=self.acknowledge_alert,
                                      bg="#cc0000", fg="white", padx=10, pady=5)
            acknowledge_btn.pack(pady=10)
            
            # Verificar si la ventana es cerrada
            self.alert_window.protocol("WM_DELETE_WINDOW", self.acknowledge_alert)
            
            # Hacer que parpadee para atraer atención
            self.blink_alert_window()
            
            # Reproducir sonido de alerta (si hay sistema de sonido disponible)
            try:
                self.alert_window.bell()
            except:
                pass
            
        except Exception as e:
            self.add_log(f"Error al mostrar notificación de alerta: {str(e)}")
    
    def blink_alert_window(self):
        """Hace parpadear la ventana de alerta"""
        if not hasattr(self, 'alert_window_active') or not self.alert_window_active:
            return
        
        if not hasattr(self, 'alert_window') or not self.alert_window.winfo_exists():
            self.alert_window_active = False
            return
        
        # Alternar color de fondo
        current_bg = self.alert_window.cget('bg')
        new_bg = "#ff6666" if current_bg == "#ffcccc" else "#ffcccc"
        self.alert_window.configure(bg=new_bg)
        
        # Programar siguiente parpadeo
        self.alert_window.after(500, self.blink_alert_window)
    
    def acknowledge_alert(self):
        """Reconoce la alerta y cierra la ventana de notificación"""
        if hasattr(self, 'alert_window') and self.alert_window.winfo_exists():
            self.alert_window.destroy()
        self.alert_window_active = False
        self.add_log("Alerta reconocida por el usuario")
    
    def export_alerts(self):
        """Exporta los datos de alertas a un archivo CSV"""
        if not self.alertas_data['timestamp']:
            messagebox.showinfo("Sin datos", "No hay alertas para exportar")
            return
        
        try:
            # Crear nombre de archivo con timestamp
            filename = f"alertas_exportadas_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Crear DataFrame
            df = pd.DataFrame(self.alertas_data)
            
            # Guardar a CSV
            df.to_csv(filename, index=False)
            
            self.add_log(f"✓ Alertas exportadas a: {filename}")
            messagebox.showinfo("Exportación Exitosa", f"Alertas guardadas en:\n{filename}")
        except Exception as e:
            self.add_log(f"❌ Error al exportar alertas: {str(e)}")
            messagebox.showerror("Error", f"No se pudieron exportar las alertas: {str(e)}")
    
    def update_charts(self):
        """Actualiza todos los gráficos con los datos recientes"""
        if not self.data['fecha']:
            return  # No hay datos para mostrar
        
        # Verificar que todas las listas de datos tienen elementos
        for key in ['temperatura_C', 'humedad_relativa', 'precipitacion_mm', 'velocidad_viento_kmh', 
                   'direccion_viento', 'luminosidad_lux', 'radiacion_solar_wm2']:
            if len(self.data[key]) == 0:
                self.add_log(f"Advertencia: No hay datos para {key}, saltando actualización de gráficas")
                return
        
        # Convertir fechas a objetos datetime para mejor visualización
        dates = []
        for d in self.data['fecha']:
            if isinstance(d, str):
                try:
                    # Intentar varios formatos de fecha
                    if '/' in d:
                        dates.append(datetime.strptime(d, "%Y/%m/%d %H:%M:%S"))
                    else:
                        dates.append(datetime.strptime(d, "%Y-%m-%d %H:%M:%S"))
                except:
                    # No registrar error para cada fecha, solo para depuración
                    dates.append(None)
            else:
                dates.append(d)
        
        # Eliminar fechas nulas y correspondientes valores de datos
        valid_indices = [i for i, date in enumerate(dates) if date is not None]
        if len(valid_indices) == 0:
            self.add_log("No hay fechas válidas para graficar")
            return
        
        cleaned_dates = [dates[i] for i in valid_indices]
        
        # Limpiar datos para cada serie de valores
        cleaned_data = {}
        for key in self.data.keys():
            if key != 'fecha':
                if len(self.data[key]) > 0:  # Verificar que hay datos
                    cleaned_data[key] = [self.data[key][i] for i in valid_indices if i < len(self.data[key])]
                else:
                    cleaned_data[key] = []
        
        # Limpiar gráficos existentes
        self.temp_ax.clear()
        self.hum_ax.clear()
        self.rain_ax.clear()
        self.wind_speed_ax.clear()
        self.wind_dir_ax.clear()
        self.rad_ax.clear()
        self.lux_ax.clear()
        
        try:
            # Verificar que hay datos suficientes para graficar
            if len(cleaned_dates) > 0 and len(cleaned_data['temperatura_C']) > 0:
                # Temperatura
                self.temp_ax.plot(cleaned_dates, cleaned_data['temperatura_C'], 'r-', marker='o', markersize=2)
                self.temp_ax.set_title('Temperatura (°C)')
                self.temp_ax.grid(True)
                self.temp_ax.set_ylabel('°C')
            
            if len(cleaned_dates) > 0 and len(cleaned_data['humedad_relativa']) > 0:
                # Humedad
                self.hum_ax.plot(cleaned_dates, cleaned_data['humedad_relativa'], 'b-', marker='o', markersize=2)
                self.hum_ax.set_title('Humedad Relativa (%)')
                self.hum_ax.grid(True)
                self.hum_ax.set_ylabel('%')
                self.hum_ax.set_xlabel('Fecha/Hora')
            
            if len(cleaned_dates) > 0 and len(cleaned_data['precipitacion_mm']) > 0:
                # Precipitación
                self.rain_ax.bar(cleaned_dates, cleaned_data['precipitacion_mm'], width=0.01, color='blue')
                self.rain_ax.set_title('Precipitación (mm)')
                self.rain_ax.grid(True)
                self.rain_ax.set_ylabel('mm')
                self.rain_ax.set_xlabel('Fecha/Hora')
            
            if len(cleaned_dates) > 0 and len(cleaned_data['velocidad_viento_kmh']) > 0:
                # Velocidad del viento
                self.wind_speed_ax.plot(cleaned_dates, cleaned_data['velocidad_viento_kmh'], 'g-', marker='o', markersize=2)
                self.wind_speed_ax.set_title('Velocidad del Viento (km/h)')
                self.wind_speed_ax.grid(True)
                self.wind_speed_ax.set_ylabel('km/h')
            
            if len(cleaned_dates) > 0 and len(cleaned_data['direccion_viento']) > 0:
                # Dirección del viento (gráfico de dispersión)
                # Convertir direcciones a valores numéricos para graficar
                dir_map = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SO': 225, 'O': 270, 'NO': 315}
                dir_values = []
                for dir_str in cleaned_data['direccion_viento']:
                    if isinstance(dir_str, str) and dir_str.upper() in dir_map:
                        dir_values.append(dir_map[dir_str.upper()])
                    else:
                        dir_values.append(None)  # Valor no válido
                        
                # Filtrar valores nulos antes de graficar
                valid_dir_indices = [i for i, val in enumerate(dir_values) if val is not None]
                if valid_dir_indices:
                    dir_dates = [cleaned_dates[i] for i in valid_dir_indices]
                    dir_values = [dir_values[i] for i in valid_dir_indices]
                    
                    self.wind_dir_ax.scatter(dir_dates, dir_values, marker='o', color='orange', s=15)
                    self.wind_dir_ax.set_title('Dirección del Viento')
                    self.wind_dir_ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315])
                    self.wind_dir_ax.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'])
                    self.wind_dir_ax.grid(True)
                    self.wind_dir_ax.set_xlabel('Fecha/Hora')
            
            if len(cleaned_dates) > 0 and len(cleaned_data['radiacion_solar_wm2']) > 0:
                # Radiación solar
                self.rad_ax.plot(cleaned_dates, cleaned_data['radiacion_solar_wm2'], 'y-', marker='o', markersize=2)
                self.rad_ax.set_title('Radiación Solar (W/m²)')
                self.rad_ax.grid(True)
                self.rad_ax.set_ylabel('W/m²')
            
            if len(cleaned_dates) > 0 and len(cleaned_data['luminosidad_lux']) > 0:
                # Luminosidad
                self.lux_ax.plot(cleaned_dates, cleaned_data['luminosidad_lux'], 'y-', marker='s', markersize=2)
                self.lux_ax.set_title('Luminosidad (lux)')
                self.lux_ax.grid(True)
                self.lux_ax.set_ylabel('lux')
                self.lux_ax.set_xlabel('Fecha/Hora')
            
            # Formato de fecha para ejes X
            for ax in [self.temp_ax, self.hum_ax, self.rain_ax, self.wind_speed_ax, 
                      self.wind_dir_ax, self.rad_ax, self.lux_ax]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            # Ajustar diseño
            self.temp_hum_fig.tight_layout()
            self.rain_fig.tight_layout()
            self.wind_fig.tight_layout()
            self.rad_fig.tight_layout()
            
            # Actualizar canvas
            self.temp_hum_canvas.draw()
            self.rain_canvas.draw()
            self.wind_canvas.draw()
            self.rad_canvas.draw()
            
            self.add_log("Gráficos actualizados con éxito")
        
        except Exception as e:
            self.add_log(f"Error al actualizar gráficos: {str(e)}")
            import traceback
            self.add_log(traceback.format_exc())
    
    def update_readings_display(self):
        """Actualiza las etiquetas con los últimos valores recibidos"""
        if not self.data['fecha']:
            return
        
        # Obtener el último índice
        idx = len(self.data['fecha']) - 1
        
        # Función auxiliar para formatear valores con seguridad
        def safe_format(value, format_str):
            try:
                if isinstance(value, str):
                    value = float(value)
                return format_str.format(value)
            except (ValueError, TypeError):
                return str(value)
        
        # Actualizar cada etiqueta con el valor más reciente
        for key in self.reading_values.keys():
            if key in self.data and idx < len(self.data[key]) and self.data[key]:
                value = self.data[key][idx]
                if key in ['temperatura_C', 'humedad_relativa', 'precipitacion_mm', 
                          'velocidad_viento_kmh', 'luminosidad_lux', 'radiacion_solar_wm2']:
                    # Formato para valores numéricos
                    display = safe_format(value, "{:.1f}")
                    unit = " °C" if key == 'temperatura_C' else \
                           " %" if key == 'humedad_relativa' else \
                           " mm" if key == 'precipitacion_mm' else \
                           " km/h" if key == 'velocidad_viento_kmh' else \
                           " lux" if key == 'luminosidad_lux' else \
                           " W/m²" if key == 'radiacion_solar_wm2' else ""
                    self.reading_values[key].config(text=f"{display}{unit}")
                elif key == 'cobertura_nubes_octas':
                    self.reading_values[key].config(text=f"{value} octas")
                else:
                    # Texto directo para valores no numéricos
                    self.reading_values[key].config(text=str(value))
        
        self.add_log("Panel de lecturas actualizado con los últimos valores")

    def process_csv_data(self, csv_text):
        """Procesa los datos CSV recibidos, limpia filas en blanco y renombra columnas"""
        # Dividir por líneas y eliminar líneas vacías
        lines = [line.strip() for line in csv_text.split('\n') if line.strip()]
        
        if not lines:
            self.add_log("Error: No hay datos CSV válidos para procesar")
            return ""
        
        # Verificar si es un archivo de alertas o de datos
        is_alerts_file = any("tipo_alerta" in line.lower() for line in lines[:3])
        
        if is_alerts_file:
            self.process_alerts_csv(lines)
            return ""  # No necesitamos devolver el CSV procesado en este caso
        
        # Verificar si la primera línea tiene encabezados o son ya datos
        first_line_has_headers = False
        if lines[0].lower().startswith("fecha") or lines[0].lower().startswith("fecha_hora"):
            first_line_has_headers = True
        
        # Si no hay encabezados, configurar encabezados predeterminados
        if not first_line_has_headers:
            self.add_log("CSV sin encabezados detectado, añadiendo encabezados")
            # Contar cuántas columnas hay para determinar qué encabezados usar
            cols = lines[0].count(',') + 1
            
            # Estos son los encabezados predeterminados si no hay encabezados en el CSV
            default_headers = ["fecha_hora", "temp_dht_cal", "hum_dht_raw", "lluvia_mm", 
                            "cobertura_nubes_octas", "vel_viento_kmh", "direccion_viento", 
                            "radiacion_solar_J_m2", "radiacion_solar_wm2", "condicion_climatica"]
            
            # Asegurarse de que tenemos suficientes encabezados
            headers = default_headers[:cols]
            while len(headers) < cols:
                headers.append(f"columna_{len(headers)+1}")
        else:
            # Usar los encabezados existentes de la primera línea
            headers = [h.strip() for h in lines[0].split(',')]
        
        # Mapear encabezados según las reglas definidas
        new_headers = []
        for header in headers:
            if header in self.column_mapping:
                new_headers.append(self.column_mapping[header])
            else:
                new_headers.append(header)
        
        # Construir el nuevo contenido CSV con los encabezados mapeados
        new_csv_lines = [','.join(new_headers)]
        
        # Agregar líneas de datos, saltando la primera si contiene encabezados
        start_idx = 1 if first_line_has_headers else 0
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line:  # Ignora líneas vacías
                # Verificar si la línea parece una instrucción no-CSV (para evitar líneas de ayuda)
                if not line.startswith("fecha") and not re.match(r'^\d{4}[-/]', line):
                    if any(word in line.lower() for word in ["ayuda", "info", "set ", "reset", "logout", "inicio_csv", "fin_csv"]):
                        continue  # Saltar esta línea, parece un comando
                new_csv_lines.append(line)
        
        return '\n'.join(new_csv_lines)
    
    def process_alerts_csv(self, lines):
        """Procesa el CSV de alertas recibido de forma más directa"""
        try:
            # Limpieza inicial de las líneas
            clean_lines = []
            for line in lines:
                # Eliminar prefijos como "CSV[1]:"
                if "CSV[" in line and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                
                # Añadir solo líneas que parezcan datos CSV
                if "," in line and not ("INICIO" in line or "FIN" in line):
                    clean_lines.append(line)
            
            # Verificar si tenemos líneas para procesar
            if not clean_lines:
                self.add_log("No se encontraron líneas de datos de alerta para procesar")
                return
            
            # Reiniciar datos de alertas
            for key in self.alertas_data:
                self.alertas_data[key] = []
            
            # Asumir que la primera línea es encabezado
            headers = [h.strip() for h in clean_lines[0].split(',')]
            
            # Procesar datos
            for i in range(1, len(clean_lines)):  # Empezar desde la segunda línea
                values = clean_lines[i].split(',')
                
                # Asegurarse de que hay suficientes valores
                if len(values) >= 7:  # Esperamos al menos 7 columnas
                    self.alertas_data['timestamp'].append(values[0].strip())
                    self.alertas_data['tipo_alerta'].append(values[1].strip())
                    
                    # Manejar la descripción que puede contener comas (está entre comillas)
                    desc = values[2].strip()
                    if desc.startswith('"') and not desc.endswith('"'):
                        # Buscar el cierre de comillas
                        desc_parts = [desc]
                        j = 3
                        while j < len(values) and not desc_parts[-1].endswith('"'):
                            desc_parts.append(values[j])
                            j += 1
                        desc = ','.join(desc_parts)
                        
                        # Ajustar los índices para los valores restantes
                        vol_idx = j
                    else:
                        vol_idx = 3
                    
                    # Limpiar comillas de la descripción
                    desc = desc.replace('"', '')
                    self.alertas_data['descripcion'].append(desc)
                    
                    # Capturar el resto de valores
                    if vol_idx < len(values):
                        self.alertas_data['volumen_l'].append(values[vol_idx].strip())
                    else:
                        self.alertas_data['volumen_l'].append("0")
                        
                    if vol_idx+1 < len(values):
                        self.alertas_data['lluvia_mm_h'].append(values[vol_idx+1].strip())
                    else:
                        self.alertas_data['lluvia_mm_h'].append("0")
                        
                    if vol_idx+2 < len(values):
                        self.alertas_data['riesgo'].append(values[vol_idx+2].strip())
                    else:
                        self.alertas_data['riesgo'].append("0")
                        
                    if vol_idx+3 < len(values):
                        self.alertas_data['humedad'].append(values[vol_idx+3].strip())
                    else:
                        self.alertas_data['humedad'].append("0")
            
            # Para depuración
            self.add_log(f"Líneas procesadas: {len(clean_lines)}")
            self.add_log(f"Registros extraídos: {len(self.alertas_data['timestamp'])}")
            
            # Mostrar primera alerta para depuración
            if self.alertas_data['timestamp']:
                self.add_log(f"Primera alerta: {self.alertas_data['timestamp'][0]}, Tipo: {self.alertas_data['tipo_alerta'][0]}")
            
            # Actualizar la tabla de alertas
            self.apply_alert_filters()
            
            # Actualizar gráficos de riesgo
            self.update_risk_charts()
            
            # Actualizar el estado actual del sistema con los valores más recientes
            self.update_current_system_status()
            
            # Verificar si hay alertas activas recientes
            self.check_for_alerts()
            
            self.add_log(f"✓ Datos de alertas procesados: {len(self.alertas_data['timestamp'])} registros")
            
            # Habilitar la pestaña de alertas
            self.notebook.tab(self.alerts_tab, state="normal")
            
            # Programar la próxima actualización automática de alertas (cada 5 minutos en lugar de 30 segundos)
            if self.connected:
                if hasattr(self, 'alert_update_job') and self.alert_update_job:
                    self.root.after_cancel(self.alert_update_job)
                
                # Verificar si hay comunicaciones en curso antes de programar
                if not self.is_communicating:
                    self.alert_update_job = self.root.after(300000, lambda: self.send_predefined_command("ALERTAS"))
                    self.add_log("Próxima actualización de alertas en 5 minutos")
                else:
                    # Si hay comunicaciones, programar verificación posterior
                    self.root.after(10000, lambda: self.resume_automatic_updates())
                    self.add_log("Programando verificación para reanudar actualizaciones posteriormente")
                
        except Exception as e:
            self.add_log(f"Error al procesar datos de alertas: {str(e)}")
            import traceback
            self.add_log(traceback.format_exc())
    def update_current_system_status(self):
        """Actualiza los indicadores del estado actual del sistema con datos de la última alerta"""
        if not self.alertas_data['timestamp']:
            return  # No hay datos para mostrar
        
        # Buscar la alerta más reciente (que no sea de tipo FIN)
        latest_alert_idx = None
        for i in range(len(self.alertas_data['timestamp'])-1, -1, -1):
            if self.alertas_data['tipo_alerta'][i] != "FIN":
                latest_alert_idx = i
                break
        
        if latest_alert_idx is None:
            return  # No se encontró alerta activa
        
        # Extraer datos de la alerta más reciente
        try:
            # Obtener valores
            tipo_alerta = self.alertas_data['tipo_alerta'][latest_alert_idx]
            volumen = float(self.alertas_data['volumen_l'][latest_alert_idx])
            lluvia = float(self.alertas_data['lluvia_mm_h'][latest_alert_idx])
            riesgo = float(self.alertas_data['riesgo'][latest_alert_idx])
            
            # Capacidad del sistema (usar valor configurado)
            capacidad = float(self.config_capacity_var.get())
            
            # Actualizar texto del estado
            if tipo_alerta.startswith("_"):
                self.alert_state_text.configure(text=f"ALERTA ACTIVA: {tipo_alerta}")
            else:
                if tipo_alerta == "MONITOR":
                    self.alert_state_text.configure(text=f"ALERTA ACTIVA: {tipo_alerta}")
                else:
                    self.alert_state_text.configure(text=f"Estado: {tipo_alerta}")
            
            # Actualizar valores numéricos
            self.volume_text.config(text=f"{volumen:.1f} L / {capacidad:.1f} L")
            self.rain_rate_text.config(text=f"{lluvia:.1f} mm/h")
            self.risk_level_text.config(text=f"{riesgo:.1f}%")
            
            # Actualizar color del indicador de estado
            if tipo_alerta.startswith("_") or tipo_alerta == "MONITOR":
                self.alert_state_canvas.configure(bg="#e74c3c")  # Rojo para alertas
            elif tipo_alerta == "FIN":
                self.alert_state_canvas.configure(bg="#2ecc71")  # Verde para normal
            else:
                self.alert_state_canvas.configure(bg="#f39c12")  # Amarillo para otros estados
            
            # Actualizar título general
            if tipo_alerta.startswith("_") or tipo_alerta == "MONITOR":
                self.alerts_title.configure(text=f"Sistema de Alertas - ¡ALERTA ACTIVA!")
                # También actualizar indicador del dashboard
                self.alert_indicator.configure(bg="#e74c3c")
                self.alert_status_label.configure(text=f"¡ALERTA! {tipo_alerta}")
            else:
                self.alerts_title.configure(text="Sistema de Alertas - Monitoreo")
                # También actualizar indicador del dashboard
                self.alert_indicator.configure(bg="#2ecc71" if tipo_alerta == "FIN" else "#f39c12")
                self.alert_status_label.configure(text="Normal" if tipo_alerta == "FIN" else tipo_alerta)
            
            # Actualizar medidor de riesgo
            self.draw_risk_gauge(riesgo)
            
            self.add_log(f"Estado actual actualizado: {tipo_alerta}, Riesgo: {riesgo:.1f}%, Vol: {volumen:.1f}L")
            
        except Exception as e:
            self.add_log(f"Error al actualizar estado actual: {str(e)}")
    def parse_csv_to_data(self, csv_text):
        """Analiza los datos CSV y actualiza los diccionarios de datos para gráficos"""
        try:
            # Dividir en líneas y extraer encabezados
            lines = [line.strip() for line in csv_text.split('\n') if line.strip()]
            if not lines:
                self.add_log("Error: CSV vacío")
                return False
            
            # Determinar si la primera línea son encabezados
            headers_line = lines[0]
            data_starts_at = 0
            
            # Si la primera línea tiene texto como "fecha" y no empieza con dígitos, 
            # asumimos que son encabezados
            if not headers_line[0].isdigit() and any(keyword in headers_line.lower() 
                                                for keyword in ['fecha', 'temp', 'hum']):
                headers = [h.strip() for h in headers_line.split(',')]
                data_starts_at = 1
            else:
                # Si no tiene encabezados, usamos los predeterminados
                self.add_log("CSV sin encabezados detectado")
                # Determinar número de columnas contando comas en la primera línea
                num_cols = headers_line.count(',') + 1
                
                # Crear encabezados predeterminados según el número de columnas
                default_headers = [
                    "fecha_hora", "temp_rtc", "temp_dht_raw", "temp_dht_cal", 
                    "hum_dht_raw", "hum_dht_cal", "vel_viento_kmh", "direccion_viento",
                    "valor_adc_veleta", "lluvia_mm", "lluvia_actual_mm", "radiacion_solar_J_m2",
                    "radiacion_solar_wm2", "cobertura_nubes_octas", "condicion_climatica"
                ]
                headers = default_headers[:num_cols]
            
            # Verificar si tenemos suficientes líneas de datos
            if data_starts_at >= len(lines):
                self.add_log("Error: No hay datos después de los encabezados")
                return False
            
            # Crear el mapeo de columnas
            column_indices = {}
            for i, header in enumerate(headers):
                # Para asegurar que lluvia_mm se mapee correctamente
                if header.lower() == 'lluvia_mm':
                    column_indices['precipitacion_mm'] = i
                # Mapear según las reglas
                elif header in self.column_mapping:
                    mapped_name = self.column_mapping[header]
                    column_indices[mapped_name] = i
                else:
                    column_indices[header] = i
            
            # Limpiar datos existentes
            for key in self.data:
                self.data[key] = []
            
            # Procesar líneas de datos
            valid_lines = 0
            for i in range(data_starts_at, len(lines)):
                line = lines[i].strip()
                if not line:  # Ignorar líneas vacías
                    continue
                
                # Verificar si la línea contiene ayuda o comandos (no válida para datos)
                if any(help_text in line.lower() for help_text in ["ayuda", "comando", "logout", "reset"]):
                    continue
                
                try:
                    values = [v.strip() for v in line.split(',')]
                    
                    # Verificar que tenemos suficientes valores
                    if len(values) < 3:  # Mínimo necesitamos fecha y algunos valores
                        continue
                    
                    # Añadir datos a cada columna mapeada
                    for mapped_col, orig_idx in column_indices.items():
                        if mapped_col in self.data and orig_idx < len(values):
                            value = values[orig_idx]
                            
                            # Convertir tipos para columnas numéricas
                            if mapped_col in ['temperatura_C', 'humedad_relativa', 'precipitacion_mm', 
                                            'velocidad_viento_kmh', 'luminosidad_lux', 
                                            'radiacion_solar_wm2', 'cobertura_nubes_octas']:
                                try:
                                    value = float(value)
                                except:
                                    value = None
                            
                            self.data[mapped_col].append(value)
                        elif mapped_col in self.data:
                            # Si la columna existe pero no hay datos, usar None
                            self.data[mapped_col].append(None)
                    
                    valid_lines += 1
                except Exception as e:
                    self.add_log(f"Advertencia: Error al procesar línea {i+1}: {e}")
                    continue
            
            if valid_lines == 0:
                self.add_log("Error: No se pudieron procesar líneas de datos válidas")
                return False
            
            # Actualizar gráficos y lecturas
            self.update_charts()
            self.update_readings_display()
            
            # Habilitar pestañas si hay datos
            if self.data['fecha']:
                self.notebook.tab(self.dash_tab, state="normal")
                self.notebook.tab(self.water_tab, state="normal")
                self.notebook.tab(self.alerts_tab, state="normal")
            
            self.add_log(f"✓ Datos procesados: {valid_lines} registros válidos")
            return True
            
        except Exception as e:
            self.add_log(f"❌ Error al procesar CSV: {str(e)}")
            import traceback
            self.add_log(traceback.format_exc())
            return False

    def calculate_water_volume(self):
        """Calcula el volumen de agua recolectada para un día específico"""
        # Obtener parámetros
        try:
            area = float(self.area_var.get())
            date_str = self.calc_date_var.get()
            
            # Validar formato de fecha
            try:
                selected_date = time.strptime(date_str, "%Y/%m/%d")
                selected_date_formatted = time.strftime("%Y/%m/%d", selected_date)
            except ValueError:
                messagebox.showerror("Error", "Formato de fecha inválido. Use YYYY/MM/DD")
                return
            
            # Verificar si tenemos datos para esta fecha
            if not self.data['fecha']:
                messagebox.showinfo("Sin datos", "No hay datos cargados para realizar el cálculo. Por favor descargue datos primero.")
                return
            
            # Convertir todas las fechas a objetos datetime para comparación
            datetime_dates = []
            for d in self.data['fecha']:
                if isinstance(d, str):
                    try:
                        # Intentar varios formatos de fecha
                        if '/' in d:
                            dt = datetime.strptime(d, "%Y/%m/%d %H:%M:%S")
                        else:
                            dt = datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
                        datetime_dates.append(dt)
                    except:
                        datetime_dates.append(None)
                else:
                    datetime_dates.append(d)
            
            # Filtrar datos del día seleccionado
            day_data = []
            for i, dt in enumerate(datetime_dates):
                if dt is not None:
                    # Comparar solo año, mes y día
                    if (dt.year == selected_date.tm_year and 
                        dt.month == selected_date.tm_mon and 
                        dt.day == selected_date.tm_mday):
                        
                        # Crear registro con datos relevantes
                        if i < len(self.data['precipitacion_mm']):
                            record = {
                                'datetime': dt,
                                'hora': dt.hour,
                                'minuto': dt.minute,
                                'precipitacion': self.data['precipitacion_mm'][i]
                            }
                            day_data.append(record)
            
            # Verificar si hay datos para ese día
            if not day_data:
                messagebox.showinfo("Sin datos", f"No se encontraron datos para la fecha: {selected_date_formatted}")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"No hay datos disponibles para el día {selected_date_formatted}\n")
                self.result_precip.config(text="0.0 mm")
                self.result_volume.config(text="0.0 L")
                
                # Limpiar gráfico de precipitación
                self.rain_day_ax.clear()
                self.rain_day_ax.set_title(f'Sin datos de precipitación para {selected_date_formatted}')
                self.rain_day_ax.grid(True)
                self.rain_day_fig.tight_layout()
                self.rain_day_canvas.draw()
                
                return
            
            # Ordenar por hora y minuto
            day_data.sort(key=lambda x: (x['hora'], x['minuto']))
            
            # Encontrar el valor máximo de precipitación del día
            max_precipitacion = 0
            max_record = None
            for record in day_data:
                if record['precipitacion'] is not None and record['precipitacion'] > max_precipitacion:
                    max_precipitacion = record['precipitacion']
                    max_record = record
            
            if max_record is None:
                max_record = day_data[-1]  # Usar el último registro si no se encontró un máximo válido
                max_precipitacion = max_record['precipitacion'] if max_record['precipitacion'] is not None else 0
            
            # Calcular volumen de agua
            volumen_litros = max_precipitacion * area
            
            # Mostrar resultados
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"CÁLCULO DE AGUA RECOLECTADA - ESTACIÓN LESTOMA\n")
            self.result_text.insert(tk.END, f"Fecha: {selected_date_formatted}\n")
            self.result_text.insert(tk.END, f"Área de recolección: {area} m²\n\n")
            
            self.result_text.insert(tk.END, f"Registros del día: {len(day_data)}\n")
            self.result_text.insert(tk.END, f"Lectura máxima: {max_record['datetime'].strftime('%H:%M:%S')}\n")
            self.result_text.insert(tk.END, f"Precipitación acumulada: {max_precipitacion:.2f} mm\n\n")
            
            self.result_text.insert(tk.END, f"RESULTADOS:\n")
            self.result_text.insert(tk.END, f"- Volumen de agua recolectada: {volumen_litros:.2f} litros\n")
            
            # Actualizar etiquetas destacadas
            self.result_precip.config(text=f"{max_precipitacion:.2f} mm")
            self.result_volume.config(text=f"{volumen_litros:.2f} L")
            
            # Actualizar visualización de agua
            self.draw_water_visualization(volumen_litros)
            
            # Mostrar tabla de datos del día
            self.result_text.insert(tk.END, "\nREGISTROS DEL DÍA:\n")
            self.result_text.insert(tk.END, "Hora      | Precipitación (mm)\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            
            for record in day_data:
                hora_str = record['datetime'].strftime('%H:%M:%S')
                precipitacion = record['precipitacion'] if record['precipitacion'] is not None else 0
                self.result_text.insert(tk.END, f"{hora_str} | {precipitacion:.2f} mm\n")
            
            # Actualizar gráfico de precipitación
            self.update_rain_day_chart(day_data, selected_date_formatted)
            
            self.add_log(f"✓ Cálculo completado para {selected_date_formatted}: {volumen_litros:.2f} litros recolectados")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo: {str(e)}")
            import traceback
            self.add_log(traceback.format_exc())
    
    def draw_water_visualization(self, volumen_litros):
        """Dibuja una visualización del agua recolectada"""
        # Limpiar canvas
        self.water_canvas.delete("all")
        
        # Configuración
        canvas_width = self.water_canvas.winfo_width()
        canvas_height = self.water_canvas.winfo_height()
        
        if canvas_width < 50 or canvas_height < 50:  # Si el canvas aún no tiene dimensiones adecuadas
            canvas_width = 200
            canvas_height = 150
        
        # Dibujar depósito
        tank_width = min(canvas_width - 80, 150)  # Más pequeño para dejar espacio para etiquetas
        tank_height = canvas_height - 40
        tank_x = (canvas_width - tank_width) // 2
        tank_y = 20
        
        # Dibujar depósito (recipiente)
        self.water_canvas.create_rectangle(tank_x, tank_y, 
                                          tank_x + tank_width, tank_y + tank_height,
                                          outline="#333333", width=2)
        
        # Calcular nivel de agua (máximo 100 litros para visualización)
        max_volume = 100  # Litros para llenar el depósito
        fill_ratio = min(1.0, volumen_litros / max_volume)
        water_height = tank_height * fill_ratio
        
        # Dibujar agua (desde abajo)
        water_y = tank_y + tank_height - water_height
        self.water_canvas.create_rectangle(tank_x + 2, water_y,
                                          tank_x + tank_width - 2, tank_y + tank_height - 2,
                                          fill="#3498db", outline="#2980b9")
        
        # Mostrar volumen como texto en el centro del agua
        text_x = tank_x + tank_width // 2
        text_y = tank_y + tank_height - (water_height / 2) if water_height > 20 else tank_y + tank_height + 10
        self.water_canvas.create_text(text_x, text_y, 
                                     text=f"{volumen_litros:.1f} L",
                                     fill="#ffffff" if fill_ratio > 0.3 else "#000000",
                                     font=("Segoe UI", 12, "bold"))
        
        # Mostrar título
        self.water_canvas.create_text(canvas_width//2, 10, 
                                     text="Agua Recolectada",
                                     font=("Segoe UI", 10, "bold"))
        
        # Agregar marcas de escala cada 20 litros
        scale_steps = 5
        for i in range(scale_steps + 1):
            volume = i * (max_volume / scale_steps)
            y_pos = tank_y + tank_height - (i / scale_steps) * tank_height
            
            # Línea de escala
            line_length = 8 if i % scale_steps == 0 or i == scale_steps else 5
            self.water_canvas.create_line(tank_x - line_length, y_pos, tank_x, y_pos, fill="#333333")
            
            # Etiqueta con el volumen
            if i % (scale_steps//2) == 0 or i == scale_steps:  # Mostrar etiquetas solo cada 40L
                self.water_canvas.create_text(tank_x - line_length - 15, y_pos,
                                            text=f"{volume:.0f} L", anchor="e",
                                            font=("Segoe UI", 8))
    
    def update_rain_day_chart(self, day_data, date_str):
        """Actualiza el gráfico de precipitación para el día seleccionado con eje X de 24 horas"""
        # Limpiar gráfico existente
        self.rain_day_ax.clear()
        
        # Preparar datos para el gráfico
        horas = [record['datetime'] for record in day_data]
        precipitacion = [record['precipitacion'] if record['precipitacion'] is not None else 0 for record in day_data]
        
        # Crear gráfico de barras para precipitación
        self.rain_day_ax.bar(horas, precipitacion, width=0.01, color='#3498db', alpha=0.7)
        self.rain_day_ax.plot(horas, precipitacion, 'o-', color='#2980b9', alpha=0.8, linewidth=1)
        
        # Configurar título y etiquetas
        self.rain_day_ax.set_title(f'Precipitación del {date_str}')
        self.rain_day_ax.set_ylabel('Precipitación (mm)')
        self.rain_day_ax.set_xlabel('Hora')
        
        # Crear un rango de 24 horas para el eje X
        start_date = datetime.strptime(date_str, "%Y/%m/%d")
        hours_range = [start_date.replace(hour=i, minute=0, second=0) for i in range(24)]
        next_day = start_date.replace(hour=0, minute=0, second=0) + pd.Timedelta(days=1)
        
        # Configurar el eje X para mostrar las 24 horas del día
        self.rain_day_ax.set_xlim(hours_range[0], next_day)
        
        # Configurar formato de hora en eje X
        hour_fmt = mdates.DateFormatter('%H:%M')
        self.rain_day_ax.xaxis.set_major_formatter(hour_fmt)
        
        # Establecer marcas principales cada 2 horas
        hour_loc = mdates.HourLocator(interval=2)
        self.rain_day_ax.xaxis.set_major_locator(hour_loc)
        
        # Añadir cuadrícula
        self.rain_day_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ajustar diseño
        self.rain_day_fig.tight_layout()
        
        # Actualizar canvas
        self.rain_day_canvas.draw()
    
    def export_water_calculation(self):
        """Exporta los resultados del cálculo de agua a un archivo"""
        if not self.result_text.get(1.0, tk.END).strip():
            messagebox.showinfo("Sin datos", "No hay resultados para exportar")
            return
        
        try:
            # Crear nombre de archivo con fecha
            date_str = self.calc_date_var.get().replace('/', '')
            filename = f"agua_recolectada_{date_str}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.result_text.get(1.0, tk.END))
            
            self.add_log(f"✓ Resultados guardados en: {filename}")
            messagebox.showinfo("Exportación Exitosa", f"Resultados guardados en:\n{filename}")
        except Exception as e:
            self.add_log(f"❌ Error al exportar resultados: {str(e)}")
            messagebox.showerror("Error", f"No se pudieron guardar los resultados: {str(e)}")
    
    def download_day_data(self):
        """Descarga los datos del día seleccionado para el cálculo"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        try:
            date_str = self.calc_date_var.get()
            
            # Validar formato de fecha
            try:
                selected_date = time.strptime(date_str, "%Y/%m/%d")
                date_str = time.strftime("%Y/%m/%d", selected_date)
            except ValueError:
                messagebox.showerror("Error", "Formato de fecha inválido. Use YYYY/MM/DD")
                return
            
            # Enviar comando para obtener datos del día específico
            self.add_log(f"Solicitando datos para el día: {date_str}")
            
            # Usar el mismo día como fecha inicio y fin
            cmd = f"DATOS {date_str} {date_str}"
            threading.Thread(target=self._send_command_thread, args=(cmd,), daemon=True).start()
            
            # Notificar al usuario
            self.add_log("Una vez recibidos los datos, puede calcular el agua recolectada")
            messagebox.showinfo("Descarga iniciada", f"Solicitando datos para {date_str}.\nUna vez recibidos, presione 'CALCULAR AGUA RECOLECTADA'.")
            
        except Exception as e:
            self.add_log(f"❌ Error al solicitar datos: {str(e)}")
            messagebox.showerror("Error", f"No se pudo iniciar la descarga: {str(e)}")

    def export_original_data(self):
        """Exporta los datos originales a un archivo CSV, eliminando líneas en blanco"""
        if hasattr(self, 'original_csv_data') and self.original_csv_data:
            try:
                # Método más agresivo para eliminar líneas vacías
                # 1. Dividir por cualquier tipo de salto de línea
                import re
                # Dividir por cualquier combinación de saltos de línea
                lines = re.split(r'\r\n|\r|\n', self.original_csv_data)
                
                # 2. Filtrar líneas vacías
                clean_lines = [line for line in lines if line.strip()]
                
                # 3. Unir con saltos de línea unificados
                clean_csv = '\n'.join(clean_lines)
                
                # 4. Verificar si hay líneas vacías después del procesamiento
                if '\n\n' in clean_csv:
                    self.add_log("Advertencia: Aún se detectan líneas vacías. Aplicando limpieza adicional.")
                    clean_csv = re.sub(r'\n\s*\n', '\n', clean_csv)
                
                filename = f"datos_original_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                
                # 5. Escribir línea por línea para mayor control
                with open(filename, 'w', encoding='utf-8') as f:
                    for line in clean_csv.split('\n'):
                        if line.strip():  # Verificación adicional
                            f.write(line + '\n')
                
                self.add_log(f"✓ Datos originales guardados en: {filename}")
                messagebox.showinfo("Exportación Exitosa", f"Datos originales guardados en:\n{filename}")
            except Exception as e:
                self.add_log(f"❌ Error al exportar datos originales: {str(e)}")
                messagebox.showerror("Error", f"No se pudieron exportar los datos: {str(e)}")
        else:
            messagebox.showinfo("Información", "No hay datos originales para exportar")

    def export_processed_data(self):
        """Exporta los datos procesados con formato específico a un archivo CSV"""
        if not self.data['fecha']:
            messagebox.showinfo("Información", "No hay datos para exportar")
            return
        
        try:
            # Crear DataFrame solo con las columnas específicas y en el orden solicitado
            required_columns = [
                'fecha', 'temperatura_C', 'humedad_relativa', 'precipitacion_mm',
                'cobertura_nubes_octas', 'velocidad_viento_kmh', 'luminosidad_lux',
                'radiacion_solar_wm2', 'direccion_viento', 'condicion_climatica'
            ]
            
            # Verificar qué columnas están disponibles
            data_dict = {}
            for col in required_columns:
                if col in self.data and self.data[col]:
                    data_dict[col] = self.data[col]
                else:
                    self.add_log(f"Advertencia: Columna '{col}' no disponible - usando valores vacíos")
                    data_dict[col] = [None] * len(self.data['fecha'])
            
            df = pd.DataFrame(data_dict)
            
            # Asegurar el orden de las columnas
            df = df[required_columns]
            
            # Guardar archivo
            filename = f"datos_procesados_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            self.add_log(f"✓ Datos procesados guardados en: {filename}")
            messagebox.showinfo("Exportación Exitosa", f"Datos procesados guardados en:\n{filename}")
            
        except Exception as e:
            self.add_log(f"❌ Error al exportar datos procesados: {str(e)}")
            messagebox.showerror("Error", f"No se pudieron exportar los datos: {str(e)}")

    def download_data_by_range(self):
        """Abre un diálogo para descargar datos en rangos específicos"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        # Crear ventana para seleccionar rango de fechas
        range_window = tk.Toplevel(self.root)
        range_window.title("Descargar Datos por Rango")
        range_window.geometry("450x280")
        range_window.resizable(False, False)
        range_window.transient(self.root)  # Hacer modal
        range_window.grab_set()
        
        # Crear campos
        frame = ttk.Frame(range_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Seleccione Rango de Fechas", 
                 font=('Segoe UI', 12, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        
        ttk.Label(frame, text="Fecha Inicio (YYYY/MM/DD):", font=('Segoe UI', 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        start_date_var = tk.StringVar(value=time.strftime("%Y/%m/%d", time.localtime(time.time() - 7*24*60*60)))  # 7 días atrás por defecto
        ttk.Entry(frame, textvariable=start_date_var, width=15, font=('Segoe UI', 10)).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Fecha Fin (YYYY/MM/DD):", font=('Segoe UI', 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        end_date_var = tk.StringVar(value=time.strftime("%Y/%m/%d"))
        ttk.Entry(frame, textvariable=end_date_var, width=15, font=('Segoe UI', 10)).grid(row=2, column=1, padx=5, pady=5)
        
        # Opciones de exportación
        export_frame = ttk.LabelFrame(frame, text="Opciones de Exportación", padding=10)
        export_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)
        
        export_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_frame, text="Exportar automáticamente al recibir", 
                      variable=export_var).pack(anchor=tk.W, pady=5)
        
        # Función para enviar comando de rango
        def send_date_range():
            start_date = start_date_var.get().strip()
            end_date = end_date_var.get().strip()
            
            # Validar formato
            try:
                time.strptime(start_date, "%Y/%m/%d")
                time.strptime(end_date, "%Y/%m/%d")
            except ValueError:
                messagebox.showerror("Error", "Formato inválido. Use YYYY/MM/DD")
                return
            
            # Guardar preferencias para la exportación
            do_export = export_var.get()
            
            # Enviar comando
            cmd = f"DATOS {start_date} {end_date}"
            self.add_log(f"Solicitando datos por rango: {cmd}")
            
            # Variable global para indicar que estamos esperando datos para exportar
            self.waiting_for_data_export = {
                'active': do_export,
                'start_date': start_date,
                'end_date': end_date
            }
            
            # Enviar comando para obtener datos
            threading.Thread(target=self._send_command_thread, args=(cmd,), daemon=True).start()
            range_window.destroy()
            
            # Si exportación automática está activada, programar verificación
            if do_export:
                self.add_log("Exportación automática activada. Se guardará cuando se reciban los datos.")
                self.check_data_received()
        
        # Botones
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Descargar", command=send_date_range, 
                  width=18, style='Big.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=range_window.destroy, 
                  width=10).pack(side=tk.LEFT, padx=5)
    
    def check_data_received(self):
        """Verifica si los datos han sido recibidos para exportación automática"""
        if hasattr(self, 'waiting_for_data_export') and self.waiting_for_data_export.get('active', False):
            if hasattr(self, 'original_csv_data') and self.original_csv_data:
                # Datos recibidos, proceder con exportación
                start_date = self.waiting_for_data_export.get('start_date', '')
                end_date = self.waiting_for_data_export.get('end_date', '')
                range_info = f"_{start_date.replace('/', '')}_{end_date.replace('/', '')}"
                
                try:
                    # Exportar datos originales
                    filename = f"datos_original{range_info}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(self.original_csv_data)
                    self.add_log(f"✓ Datos originales guardados en: {filename}")
                    messagebox.showinfo("Exportación Exitosa", f"Datos originales guardados en:\n{filename}")
                    
                    # Exportar datos procesados
                    filename = f"datos_procesados{range_info}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    required_columns = [
                        'fecha', 'temperatura_C', 'humedad_relativa', 'precipitacion_mm',
                        'cobertura_nubes_octas', 'velocidad_viento_kmh', 'luminosidad_lux',
                        'radiacion_solar_wm2', 'direccion_viento', 'condicion_climatica'
                    ]
                    
                    data_dict = {}
                    for col in required_columns:
                        if col in self.data and self.data[col]:
                            data_dict[col] = self.data[col]
                        else:
                            data_dict[col] = [None] * len(self.data['fecha'])
                    
                    df = pd.DataFrame(data_dict)
                    df = df[required_columns]
                    df.to_csv(filename, index=False)
                    
                    self.add_log(f"✓ Datos procesados guardados en: {filename}")
                    messagebox.showinfo("Exportación Exitosa", f"Datos procesados guardados en:\n{filename}")
                except Exception as e:
                    self.add_log(f"❌ Error en exportación automática: {str(e)}")
                
                # Limpiar flag
                self.waiting_for_data_export = {'active': False}
            else:
                # Si aún no hay datos, programar otra verificación en 2 segundos
                self.root.after(2000, self.check_data_received)

    def refresh_ports(self):
        """Buscar puertos COM disponibles"""
        self.add_log("Buscando puertos COM...")
        
        ports = list(serial.tools.list_ports.comports())
        port_list = []
        
        for port in ports:
            description = port.description.lower()
            # Destacar posibles puertos Bluetooth
            if "bluetooth" in description or "bt" in description or "serial" in description:
                port_list.append(f"→ {port.device} - {port.description} [POSIBLE ESTACIÓN]")
            else:
                port_list.append(f"{port.device} - {port.description}")
        
        self.port_combo['values'] = port_list
        
        if port_list:
            self.port_combo.current(0)
            self.add_log(f"✓ Se encontraron {len(port_list)} puertos COM")
            
            # Buscar puerto con "Bluetooth" en su nombre para seleccionarlo automáticamente
            bluetooth_index = -1
            for i, port_str in enumerate(port_list):
                if "bluetooth" in port_str.lower() or "bt" in port_str.lower():
                    bluetooth_index = i
                    break
            
            if bluetooth_index >= 0:
                self.port_combo.current(bluetooth_index)
                self.add_log(f"✓ Seleccionado automáticamente puerto Bluetooth: {port_list[bluetooth_index]}")
        else:
            self.add_log("❌ No se encontraron puertos COM disponibles")
    
    def add_log(self, message):
        """Añade un mensaje al log con timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Auto-scroll al final
    
    def connect(self):
        """Conectar/Desconectar del puerto seleccionado"""
        if not self.connected:
            # Obtener puerto seleccionado
            selected_port = self.port_var.get()
            if not selected_port:
                messagebox.showerror("Error", "Seleccione un puerto COM primero")
                return
            
            # Extraer nombre del puerto COM correctamente
            # Buscar 'COM' en la cadena seleccionada
            if "COM" in selected_port:
                # Encontrar la posición donde aparece 'COM'
                com_pos = selected_port.find("COM")
                # Extraer el puerto completo (COM más el número)
                port = ""
                for i in range(com_pos, len(selected_port)):
                    if selected_port[i] == ' ' and port.startswith("COM"):
                        break
                    port += selected_port[i]
            else:
                messagebox.showerror("Error", "No se pudo identificar un puerto COM válido")
                return
            
            # Mostrar puerto extraído para depuración
            self.add_log(f"Puerto extraído: '{port}'")
            
            # Obtener baudrate y contraseña
            baudrate = int(self.baudrate_var.get())
            password = self.password_var.get()
            
            self.add_log(f"⏳ Conectando a {port} a {baudrate} baudios...")
            self.connect_btn.configure(state=tk.DISABLED)
            
            # Iniciar conexión en hilo separado
            threading.Thread(target=self._connect_thread, 
                            args=(port, baudrate, password), 
                            daemon=True).start()
        else:
            # Desconectar
            self._disconnect()
    
    def _connect_thread(self, port, baudrate, password):
        """Proceso de conexión en hilo separado"""
        try:
            # Cerrar conexión previa si existe
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.serial_conn = None
                time.sleep(1)  # Dar tiempo al sistema para liberar el puerto
            
            # Intentar conectar 3 veces
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    self.root.after(0, lambda: self.add_log(f"Intento de conexión {attempt+1}/{max_attempts}..."))
                    self.serial_conn = serial.Serial(port, baudrate, timeout=2)
                    break
                except serial.SerialException as e:
                    if attempt < max_attempts - 1:
                        self.root.after(0, lambda: self.add_log(f"❌ Intento fallido: {str(e)}"))
                        time.sleep(2)
                    else:
                        raise e
            
            if not self.serial_conn or not self.serial_conn.is_open:
                raise Exception("No se pudo establecer conexión serial")
                
            self.root.after(0, lambda: self.add_log("✓ Conexión serial establecida. Intentando autenticación..."))
            
            # Proceso de autenticación mejorado
            time.sleep(1)
            
            # 1. Enviar varios enters para provocar respuesta
            for _ in range(3):
                self.serial_conn.write(b"\r\n")
                time.sleep(0.5)
            
            # 2. Limpiar buffer de entrada
            if self.serial_conn.in_waiting:
                data = self.serial_conn.read(self.serial_conn.in_waiting)
                self.root.after(0, lambda: self.add_log(f"Respuesta inicial: {data.decode('utf-8', errors='replace')}"))
            
            # 3. Enviar contraseña directamente
            self.root.after(0, lambda: self.add_log(f"Enviando contraseña: {password}"))
            self.serial_conn.write(f"{password}\r\n".encode())
            
            # 4. Esperar respuesta con tiempo suficiente
            time.sleep(2)
            auth_response = ""
            timeout = time.time() + 10  # 10 segundos para respuesta
            
            while time.time() < timeout:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='replace')
                    auth_response += line
                    self.root.after(0, lambda l=line: self.add_log(f"< {l.strip()}"))
                    
                    # Buscar palabras clave en respuesta
                    if any(keyword in line.lower() for keyword in ["exitosa", "bienvenido", "ayuda", "comando"]):
                        self.root.after(0, lambda: self.add_log("✅ Autenticación exitosa!"))
                        self.root.after(0, self._update_ui_connected)
                        # Habilitar pestañas
                        self.root.after(0, lambda: self.notebook.tab(self.dash_tab, state="normal"))
                        self.root.after(0, lambda: self.notebook.tab(self.water_tab, state="normal"))
                        self.root.after(0, lambda: self.notebook.tab(self.alerts_tab, state="normal"))
                        # Solicitar comandos disponibles
                        time.sleep(0.5)
                        self.serial_conn.write(b"AYUDA\r\n")
                        return
                
                time.sleep(0.2)
                
                # Si hay alguna respuesta pero no es explícita, consideramos éxito
                if auth_response and time.time() > timeout - 8:
                    if not any(error in auth_response.lower() for error in ["error", "incorrecta", "inválida"]):
                        self.root.after(0, lambda: self.add_log("✅ Conexión establecida (respuesta implícita)"))
                        self.root.after(0, self._update_ui_connected)
                        # Habilitar pestañas
                        self.root.after(0, lambda: self.notebook.tab(self.dash_tab, state="normal"))
                        self.root.after(0, lambda: self.notebook.tab(self.water_tab, state="normal"))
                        self.root.after(0, lambda: self.notebook.tab(self.alerts_tab, state="normal"))
                        return
            
            # 5. Reintento si falló la primera vez
            if "contraseña" in auth_response.lower() or auth_response.strip() == "":
                self.root.after(0, lambda: self.add_log("⏳ Reintentando autenticación..."))
                self.serial_conn.write(f"{password}\r\n".encode())
                
                time.sleep(2)
                retry_response = ""
                timeout = time.time() + 5
                
                while time.time() < timeout:
                    if self.serial_conn.in_waiting:
                        line = self.serial_conn.readline().decode('utf-8', errors='replace')
                        retry_response += line
                        self.root.after(0, lambda l=line: self.add_log(f"< {l.strip()}"))
                
                if any(keyword in retry_response.lower() for keyword in ["exitosa", "bienvenido", "ayuda", "comando"]):
                    self.root.after(0, lambda: self.add_log("✅ Autenticación exitosa en segundo intento!"))
                    self.root.after(0, self._update_ui_connected)
                    # Habilitar pestañas
                    self.root.after(0, lambda: self.notebook.tab(self.dash_tab, state="normal"))
                    self.root.after(0, lambda: self.notebook.tab(self.water_tab, state="normal"))
                    self.root.after(0, lambda: self.notebook.tab(self.alerts_tab, state="normal"))
                    return
            
            # Si llegamos aquí, falló la autenticación
            raise Exception(f"Autenticación fallida. Respuesta: {auth_response}")
            
        except Exception as e:
            self.root.after(0, lambda: self.add_log(f"❌ ERROR: {str(e)}"))
            self.root.after(0, self._update_ui_disconnected)
            
            # Cerrar conexión en caso de error
            if self.serial_conn:
                try:
                    self.serial_conn.close()
                except:
                    pass
                self.serial_conn = None
    
    def _update_ui_connected(self):
        """Actualiza la UI para mostrar estado conectado"""
        self.connected = True
        self.connect_btn.configure(text="DESCONECTAR", state=tk.NORMAL)
        self.status_indicator.configure(bg="green")
        self.status_text.configure(text="Conectado")
        
        # Cambiar a pestaña de dashboard
        self.notebook.select(self.dash_tab)
        
        # Iniciar actualización automática si está activada
        if self.auto_update_var.get():
            self.schedule_auto_update()
        
        # Iniciar actualizaciones automáticas de alertas (cada 5 minutos en vez de 30 segundos)
        if hasattr(self, 'alert_update_job') and self.alert_update_job:
            self.root.after_cancel(self.alert_update_job)
        
        self.alert_update_job = self.root.after(300000, lambda: self.send_predefined_command("ALERTAS"))
        self.add_log("Comenzando monitoreo automático de alertas (cada 5 minutos)")
    
    def _update_ui_disconnected(self):
        """Actualiza la UI para mostrar estado desconectado"""
        self.connected = False
        self.connect_btn.configure(text="CONECTAR", state=tk.NORMAL)
        self.status_indicator.configure(bg="red")
        self.status_text.configure(text="Desconectado")
        
        # Desactivar pestañas
        self.notebook.tab(self.dash_tab, state="disabled")
        self.notebook.tab(self.water_tab, state="disabled")
        self.notebook.tab(self.alerts_tab, state="disabled")
        
        # Cancelar actualizaciones automáticas
        if self.auto_update_job:
            self.root.after_cancel(self.auto_update_job)
            self.auto_update_job = None
    
    def _disconnect(self):
        """Cerrar la conexión serial"""
        self.add_log("⏳ Desconectando...")
        
        # Desactivar actualización automática
        if self.auto_update_var.get():
            self.auto_update_var.set(False)
            self.toggle_auto_update()
        
        # Cancelar actualizaciones automáticas de alertas
        if hasattr(self, 'alert_update_job') and self.alert_update_job:
            self.root.after_cancel(self.alert_update_job)
            self.alert_update_job = None
        
        # Limpiar cola de comandos
        self.command_queue = []
        self.queue_processing = False
        
        # Liberar semáforo si está bloqueado
        self.is_communicating = False
        
        if self.serial_conn:
            try:
                # Intentar hacer logout limpio
                self.serial_conn.write(b"LOGOUT\r\n")
                time.sleep(1)
                
                # Leer respuesta
                if self.serial_conn.in_waiting:
                    response = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='replace')
                    self.add_log(f"Respuesta a LOGOUT: {response}")
                
                # Cerrar conexión
                self.serial_conn.close()
                time.sleep(0.5)
            except Exception as e:
                self.add_log(f"Error al desconectar: {str(e)}")
            finally:
                self.serial_conn = None
        
        self._update_ui_disconnected()
        self.add_log("✓ Desconectado")
        
    def save_data_cache(self):
        """Guarda los datos en caché local manteniendo el formato actual"""
        if not self.data['fecha']:
            return False
            
        cache_file = "estacion_data_cache.pkl"
        try:
            # Guardar diccionario completo
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
                
            # También guardar datos de alertas si están disponibles
            if self.alertas_data['timestamp']:
                with open('estacion_alertas_cache.pkl', 'wb') as f:
                    pickle.dump(self.alertas_data, f)
                
            # También guardar los datos originales si están disponibles
            if hasattr(self, 'original_csv_data') and self.original_csv_data:
                with open('estacion_original_data.csv', 'w', encoding='utf-8') as f:
                    f.write(self.original_csv_data)
                    
            self.add_log(f"✓ Datos guardados en caché local: {len(self.data['fecha'])} registros")
            return True
        except Exception as e:
            self.add_log(f"❌ Error al guardar caché: {str(e)}")
            return False
    
    def load_data_cache(self):
        """Carga datos desde caché local preservando el formato actual"""
        cache_file = "estacion_data_cache.pkl"
        if not os.path.exists(cache_file):
            self.add_log("No se encontró archivo de caché local")
            return False
            
        try:
            # Cargar diccionario completo
            import pickle
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            
            # Intentar cargar datos de alertas
            if os.path.exists('estacion_alertas_cache.pkl'):
                with open('estacion_alertas_cache.pkl', 'rb') as f:
                    self.alertas_data = pickle.load(f)
                self.add_log(f"✓ Datos de alertas cargados: {len(self.alertas_data['timestamp'])} alertas")
                self.apply_alert_filters()
                self.update_risk_charts()
            
            # Intentar cargar también los datos originales
            if os.path.exists('estacion_original_data.csv'):
                with open('estacion_original_data.csv', 'r', encoding='utf-8') as f:
                    self.original_csv_data = f.read()
                
            self.add_log(f"✓ Datos cargados desde caché local: {len(self.data['fecha'])} registros")
            
            # Actualizar gráficos y lecturas
            self.update_charts()
            self.update_readings_display()
            
            # Habilitar pestañas
            self.notebook.tab(self.dash_tab, state="normal")
            self.notebook.tab(self.water_tab, state="normal")
            self.notebook.tab(self.alerts_tab, state="normal")
            
            return True
        except Exception as e:
            self.add_log(f"❌ Error al cargar caché: {str(e)}")
            return False
    
    def send_command(self):
        """Envía el comando del campo de texto"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        command = self.cmd_var.get().strip()
        if not command:
            return
        
        self.add_log(f"Enviando: {command}")
        self.queue_command(command)
        self.cmd_var.set("")  # Limpiar campo
    
    def send_predefined_command(self, command):
        """Envía un comando predefinido"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        # Para comandos importantes como DATOS, interrumpir actualizaciones automáticas
        if command.upper().startswith("DATOS"):
            self.pause_automatic_updates(60000)  # Pausar por 60 segundos
            self.add_log(f"Actualizaciones de alertas pausadas durante 60 segundos para comandos de datos")
        
        # Añadir a la cola en lugar de enviar directamente
        self.add_log(f"Agregando comando a la cola: {command}")
        self.queue_command(command)
    def pause_automatic_updates(self, duration=60000):
        """Pausa las actualizaciones automáticas durante el tiempo especificado (ms)"""
        if hasattr(self, 'alert_update_job') and self.alert_update_job:
            self.root.after_cancel(self.alert_update_job)
            self.alert_update_job = None
            self.add_log("Actualizaciones automáticas pausadas temporalmente")
            
            # Programar la reanudación después del tiempo especificado si se proporcionó duración
            if duration > 0:
                self.root.after(duration, self.resume_automatic_updates)

    def resume_automatic_updates(self):
        """Reanuda las actualizaciones automáticas"""
        if self.connected:
            if hasattr(self, 'alert_update_job') and self.alert_update_job:
                self.root.after_cancel(self.alert_update_job)
            
            # Programar la próxima actualización de alertas (5 minutos)
            self.alert_update_job = self.root.after(300000, lambda: self.send_predefined_command("ALERTAS"))
            self.add_log("Actualizaciones automáticas reanudadas (próxima en 5 minutos)")
    def _send_command_thread(self, command):
        """Envía comando y recibe respuesta en hilo separado usando semáforo"""
        # Verificar si ya hay una comunicación en curso
        if self.is_communicating:
            self.root.after(0, lambda: self.add_log(f"⚠️ Comando {command} rechazado: comunicación en curso"))
            return
            
        self.is_communicating = True
        original_timeout = None
        
        try:
            # Limpiar buffer completamente antes de enviar
            self.serial_conn.reset_input_buffer()
            
            # Enviar comando
            self.serial_conn.write(f"{command}\r\n".encode())
            
            # Esperar un poco más antes de empezar a leer
            time.sleep(1.0)
            
            # Variables para lectura
            csv_mode = False
            csv_buffer = []
            timeout = time.time() + 120  # Tiempo extendido para datos grandes
            csv_inicio_detectado = False
            csv_fin_detectado = False
            
            # Ajustar el timeout serial para lecturas más rápidas durante el proceso
            original_timeout = self.serial_conn.timeout
            self.serial_conn.timeout = 0.5
            
            while time.time() < timeout:
                try:
                    # Leer una línea completa (hasta \n o \r\n)
                    line = self.serial_conn.readline().decode('latin-1', errors='replace').strip()
                    
                    if not line:
                        if csv_fin_detectado or (not csv_mode and time.time() > timeout - 110):
                            # Esperamos un poco más y si no hay datos, salimos
                            time.sleep(0.2)
                            if self.serial_conn.in_waiting == 0:
                                break
                        continue
                    
                    # Detectar inicio de datos CSV
                    if "INICIO_DATOS" in line:
                        csv_mode = True
                        csv_inicio_detectado = True
                        csv_buffer = []  # Reiniciar buffer
                        self.root.after(0, lambda: self.add_log("--- INICIO DE DATOS CSV ---"))
                        continue
                    
                    # Detectar fin de datos CSV
                    if "FIN_DATOS" in line and csv_mode:
                        csv_mode = False
                        csv_fin_detectado = True
                        self.root.after(0, lambda: self.add_log("--- FIN DE DATOS CSV ---"))
                        
                        # Procesar CSV completo
                        if csv_buffer:
                            try:
                                # Convertir buffer a texto CSV
                                csv_text = "\n".join(csv_buffer)
                                
                                # Guardar datos originales
                                self.original_csv_data = csv_text
                                
                                # Determinar si es una solicitud de datos de alertas
                                is_alertas = command.upper() == "ALERTAS"
                                
                                if is_alertas:
                                    # Procesar directamente como alertas
                                    self.process_alerts_csv(csv_buffer)
                                else:
                                    # Procesar para visualización
                                    processed_csv = self.process_csv_data(csv_text)
                                    self.parse_csv_to_data(processed_csv)
                                
                                # Verificar si estamos en una descarga por rango
                                if hasattr(self, 'waiting_for_data_export') and self.waiting_for_data_export.get('active', False):
                                    self.check_data_received()
                                
                                # Guardar caché automáticamente después de recibir datos completos
                                if command.upper().startswith("DATOS") or command.upper() == "ALERTAS":
                                    self.root.after(0, self.save_data_cache)
                                    
                                self.root.after(0, lambda: self.add_log(f"✓ Datos recibidos: {len(csv_buffer)} líneas"))
                            except Exception as e:
                                self.root.after(0, lambda: self.add_log(f"❌ Error procesando CSV: {str(e)}"))
                                import traceback
                                self.root.after(0, lambda: self.add_log(traceback.format_exc()))
                        continue
                    
                    # Si estamos en modo CSV, almacenar la línea en el buffer
                    if csv_mode:
                        # Verificar si la línea parece ser una línea CSV válida
                        if re.match(r'^\d{4}[-/]', line) or ('fecha' in line.lower() and ',' in line):
                            csv_buffer.append(line)
                            # Mostrar solo algunas líneas para no saturar el log
                            if len(csv_buffer) % 50 == 1 or len(csv_buffer) < 5:
                                self.root.after(0, lambda l=line: self.add_log(f"CSV[{len(csv_buffer)}]: {l[:60]}..."))
                    else:
                        # Reemplazar caracteres especiales en mensajes
                        fixed_line = line
                        fixed_line = fixed_line.replace("TransmisiÃ³n", "Transmisión")
                        fixed_line = fixed_line.replace("lÃ­neas", "líneas")
                        fixed_line = fixed_line.replace("aplicaciÃ³n", "aplicación")
                        fixed_line = fixed_line.replace("estÃ¡n", "están")
                        self.root.after(0, lambda l=fixed_line: self.add_log(f"< {l}"))
                        
                        # Si la respuesta contiene información del sistema, procesar variables de configuración
                        if command.upper() == "INFO" and any(key in line for key in ["Área", "Capacidad", "Volumen"]):
                            self.process_info_response(fixed_line)
                        
                except Exception as read_error:
                    self.root.after(0, lambda e=read_error: self.add_log(f"⚠️ Error de lectura: {str(e)}"))
                    time.sleep(0.2)  # Breve pausa antes de continuar
            
            # Restaurar timeout original
            if original_timeout is not None:
                self.serial_conn.timeout = original_timeout
            
            # Verificar si es logout
            if command.upper() == "LOGOUT":
                self.root.after(0, self._disconnect)
                
        except Exception as e:
            self.root.after(0, lambda: self.add_log(f"❌ Error general: {str(e)}"))
            import traceback
            self.root.after(0, lambda: self.add_log(traceback.format_exc()))
            
        finally:
            # Liberar semáforo siempre, sin importar si hubo éxito o error
            self.is_communicating = False
            
            # Restaurar timeout en caso de error
            if original_timeout is not None and self.serial_conn:
                self.serial_conn.timeout = original_timeout
            
            # Si es un comando de datos, reanudar las actualizaciones automáticas
            if command.upper().startswith("DATOS"):
                self.root.after(1000, self.resume_automatic_updates)
    def queue_command(self, command):
        """Añade un comando a la cola y procesa si no hay comandos en ejecución"""
        self.command_queue.append(command)
        if not self.queue_processing:
            self.process_command_queue()

    def process_command_queue(self):
        """Procesa comandos en la cola de forma secuencial"""
        if self.is_communicating:
            # Si hay una comunicación en curso, programar verificación posterior
            self.root.after(1000, self.process_command_queue)
            return
            
        if not self.command_queue:
            self.queue_processing = False
            return
            
        self.queue_processing = True
        command = self.command_queue.pop(0)
        self.add_log(f"Procesando comando en cola: {command}")
        
        # Iniciar nuevo hilo para ejecutar comando
        threading.Thread(target=self._send_command_thread, args=(command,), daemon=True).start()
        
        # Programar verificación para el siguiente comando
        self.root.after(1000, self.process_command_queue)
    def process_info_response(self, line):
        """Procesa líneas de información del sistema para extraer configuración"""
        try:
            # Extraer área de recolección
            area_match = re.search(r'Área.*?:\s*(\d+\.?\d*)', line)
            if area_match:
                area_value = float(area_match.group(1))
                self.config_area_var.set(str(area_value))
                self.area_var.set(str(area_value))  # También actualizar en pestaña de cálculo
                self.add_log(f"Área de recolección configurada: {area_value} m²")
            
            # Extraer capacidad del sistema
            capacity_match = re.search(r'[Cc]apacidad.*?:\s*(\d+\.?\d*)', line)
            if capacity_match:
                capacity_value = float(capacity_match.group(1))
                self.config_capacity_var.set(str(capacity_value))
                self.add_log(f"Capacidad del sistema configurada: {capacity_value} L")
            
            # Extraer volumen actual
            volume_match = re.search(r'[Vv]olumen.*?:\s*(\d+\.?\d*)\s*L', line)
            if volume_match:
                volume_value = float(volume_match.group(1))
                self.volume_text.config(text=f"{volume_value:.1f} L / {self.config_capacity_var.get()} L")
                self.add_log(f"Volumen actual: {volume_value:.1f} L")
                
        except Exception as e:
            self.add_log(f"Error al procesar información del sistema: {str(e)}")
    
    def set_datetime(self):
        """Configura la fecha y hora en la estación"""
        if not self.connected or not self.serial_conn:
            messagebox.showerror("Error", "No hay conexión activa")
            return
        
        # Crear ventana para fecha/hora
        date_window = tk.Toplevel(self.root)
        date_window.title("Configurar Fecha y Hora")
        date_window.geometry("400x220")
        date_window.resizable(False, False)
        date_window.transient(self.root)  # Hacer modal
        date_window.grab_set()
        
        # Obtener fecha y hora actual
        current_time = time.localtime()
        
        # Crear campos
        frame = ttk.Frame(date_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Fecha (DD/MM/YYYY):", font=('Segoe UI', 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        date_var = tk.StringVar(value=time.strftime("%d/%m/%Y", current_time))
        ttk.Entry(frame, textvariable=date_var, width=15, font=('Segoe UI', 10)).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Hora (HH:MM:SS):", font=('Segoe UI', 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        time_var = tk.StringVar(value=time.strftime("%H:%M:%S", current_time))
        ttk.Entry(frame, textvariable=time_var, width=15, font=('Segoe UI', 10)).grid(row=1, column=1, padx=5, pady=5)
        
        # Función para enviar
        def send_datetime():
            date_str = date_var.get()
            time_str = time_var.get()
            
            # Validar formato
            try:
                # Convertir del formato día/mes/año a año/mes/día para el comando
                day, month, year = date_str.split('/')
                cmd_date = f"{day}/{month}/{year}"
                
                # Validar hora
                h, m, s = map(int, time_str.split(':'))
                if h < 0 or h > 23 or m < 0 or m > 59 or s < 0 or s > 59:
                    raise ValueError("Hora inválida")
                
            except ValueError:
                messagebox.showerror("Error", "Formato inválido. Use DD/MM/YYYY y HH:MM:SS")
                return
            
            # Enviar comando
            cmd = f"SET {cmd_date} {time_str}"
            self.add_log(f"Enviando fecha/hora: {cmd}")
            threading.Thread(target=self._send_command_thread, args=(cmd,), daemon=True).start()
            date_window.destroy()
        
        # Botones
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Sincronizar y Enviar", command=send_datetime, 
                  width=18, style='Big.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=date_window.destroy, 
                 width=10).pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = EstacionMeteoApp(root)
    root.mainloop()

######## By: Bryan Rojas and Nathalia Gutierrez ########
# 2024-01-01