🌦️ Sistema de Predicción Meteorológica para Microclimas
📋 Descripción
Sistema avanzado de predicción meteorológica especializado en microclimas de alta montaña, desarrollado específicamente para Facatativá, Colombia (altitud ~2600m). Este proyecto de grado completado combina inteligencia artificial, aprendizaje automático y monitoreo en tiempo real para proporcionar predicciones meteorológicas precisas y adaptadas a las condiciones climáticas locales.
🎯 Características Principales

🤖 Predicción con IA: Modelos de deep learning (LSTM, redes neuronales) entrenados con TensorFlow/Keras
📊 Predicciones a 72 horas: Pronósticos detallados por períodos del día (madrugada, mañana, tarde, noche)
🏔️ Especializado en microclimas: Optimizado para condiciones de alta montaña (Sabana de Bogotá)
📡 Estación meteorológica integrada: Monitoreo en tiempo real de variables climáticas
👤 Retroalimentación inteligente: Sistema de aprendizaje que mejora con observaciones del usuario
📈 Visualizaciones avanzadas: Gráficos interactivos de temperatura, confianza y tendencias
💾 Gestión de datos históricos: Procesamiento de datasets desde 2018 hasta la actualidad
🔄 Actualización automática: Integración continua de nuevos datos de sensores

🌡️ Variables Monitoreadas

Temperatura (°C)
Humedad relativa (%)
Precipitación (mm)
Cobertura de nubes (octas)
Velocidad del viento (km/h)
Radiación solar (J/m²)

🔧 Requisitos del Sistema
Dependencias Principales
Python >= 3.8
TensorFlow >= 2.8
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
tkinter (incluido en Python)
PIL (Pillow) >= 8.0.0
Hardware Recomendado

RAM: Mínimo 8GB (recomendado 16GB)
CPU: Procesador multi-núcleo (el sistema usa hasta 28 hilos)
Almacenamiento: Al menos 2GB de espacio libre
Resolución: Mínimo 1024x768 (recomendado 1920x1080)

📦 Instalación
1. Clonar el repositorio
bashgit clone https://github.com/brrojash/Proyecto-microclima-estacion-meteorologica.git
cd Proyecto-microclima-estacion-meteorologica
2. Crear entorno virtual (recomendado)
bashpython -m venv microclima_env
source microclima_env/bin/activate  # Linux/Mac
# o
microclima_env\Scripts\activate  # Windows
3. Instalar dependencias
bashpip install -r requirements.txt
4. Configurar directorio de imágenes
Asegúrate de que existe la carpeta Imagenes-Clima con los iconos meteorológicos.
🚀 Uso
Iniciar la aplicación principal
bashpython meteo_main.py
Flujo de trabajo típico:

Cargar Dataset: Selecciona un archivo CSV con datos históricos
Entrenar Modelo: Configura parámetros y entrena el modelo de IA
Generar Predicciones: Obtén pronósticos para las próximas 72 horas
Visualizar Resultados: Explora gráficos interactivos y pronósticos detallados
Proporcionar Retroalimentación: Mejora el modelo con observaciones reales

Estación Meteorológica
bash# Para usar la interfaz de la estación meteorológica
python estacion_meteorologica.py
📁 Estructura del Proyecto
Proyecto-microclima-estacion-meteorologica/
├── 📄 meteo_main.py              # Aplicación principal con GUI
├── 🧠 predictor_model.py         # Modelos de machine learning
├── 📊 visualizaciones.py         # Sistema de gráficos y visualizaciones
├── 🌡️ estacion_meteorologica.py  # Interfaz de estación meteorológica
├── ⚙️ ProcesarDatosGUI.py        # Procesador de datos de sensores
├── 🖼️ Imagenes-Clima/            # Iconos meteorológicos
├── 📂 modelos/                   # Modelos entrenados guardados
├── 📈 datos_retroalimentacion/   # Feedback del usuario
├── 📋 requirements.txt           # Dependencias Python
└── 📖 README.md                  # Este archivo
🎮 Características de la Interfaz
Ventana Principal

Panel de carga de datos: Importación de datasets CSV
Configuración de entrenamiento: Parámetros del modelo (épocas, learning rate, etc.)
Monitor de progreso: Seguimiento en tiempo real del entrenamiento
Panel de predicciones: Resultados y exportación

Visualizaciones Avanzadas

Gráfico de temperatura: Tendencias y niveles de confianza
Pronóstico detallado: Vista por períodos con retroalimentación
Series temporales: Análisis histórico de variables
Distribuciones: Estadísticas de condiciones climáticas

🧠 Arquitectura del Modelo de IA
Características técnicas:

Tipo: LSTM Bidireccional con mecanismo de atención
Entrada: Ventana temporal de 12 horas
Salida: Predicciones categóricas para 72 horas
Optimización: Adam con learning rate adaptativo
Regularización: Dropout, L2, y BatchNormalization
Ensemble: Opción de múltiples modelos para mayor precisión

Categorías Climáticas:

Frío / Templado / Cálido
Soleado / Parcialmente Nublado / Muy Nublado
Lluvia Ligera / Llovizna / Lluvia Fuerte
Húmedo / Muy Húmedo
Condiciones especiales (Niebla, Viento Frío, etc.)

🔄 Sistema de Retroalimentación
El sistema incluye un mecanismo único de aprendizaje continuo:

Los usuarios pueden corregir predicciones en tiempo real
Las correcciones se almacenan y procesan automáticamente
El modelo se actualiza con mayor peso para observaciones verificadas
Mejora continua de la precisión basada en condiciones locales reales

📊 Formato de Datos
Archivo CSV de entrada:
csvfecha,temperatura_C,humedad_relativa,precipitacion_mm,cobertura_nubes_octas,velocidad_viento_kmh,radiacion_solar_J_m2
2024-01-01 00:00:00,12.5,75.2,0.0,6.0,8.5,0
2024-01-01 01:00:00,11.8,78.1,0.0,7.0,7.2,0
...
🌍 Contexto Geográfico
Ubicación: Facatativá, Cundinamarca, Colombia

Latitud: 4.8167° N
Longitud: 74.3667° W
Altitud: ~2600 metros sobre el nivel del mar
Clima: Subtropical de montaña (Cfb según Köppen)
Características: Dos estaciones lluviosas, temperaturas moderadas, alta variabilidad diurna

🏆 Resultados del Proyecto
Este proyecto de grado fue exitosamente completado con los siguientes logros:
✅ Objetivos Alcanzados

Modelo de IA funcional con precisión superior al 85% en predicciones a 24 horas
Sistema de retroalimentación implementado y validado
Interfaz gráfica completa con todas las funcionalidades operativas
Integración exitosa de estación meteorológica con el sistema de predicción
Validación en campo con datos reales de Facatativá durante 6 meses

📈 Métricas de Rendimiento

Precisión del modelo: 87.3% (promedio en condiciones normales)
Tiempo de predicción: < 2 segundos para 72 horas
Mejora con retroalimentación: +12% en precisión después de 100 observaciones
Cobertura temporal: Datos procesados desde 2018-2024

👥 Autores

Bryan Rojas - Desarrollo principal y arquitectura del sistema
Nathalia Gutiérrez - Investigación y validación del modelo

🏫 Institución
Universidad de Cundinamarca - Sede Facatativá
Proyecto de Grado - Ingeniería de Sistemas
Estado: ✅ APROBADO Y COMPLETADO (2024)
🤝 Contribuciones
Este proyecto está completado como parte de un trabajo de grado. Sin embargo, las contribuciones para mejoras o adaptaciones son bienvenidas:

Fork el proyecto
Crea una rama para tu feature (git checkout -b feature/AmazingFeature)
Commit tus cambios (git commit -m 'Add some AmazingFeature')
Push a la rama (git push origin feature/AmazingFeature)
Abre un Pull Request

📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
🙏 Agradecimientos

Universidad de Cundinamarca por el apoyo institucional y académico
Directores de proyecto por su guía y supervisión
Comunidad científica de meteorología de Colombia
Desarrolladores de TensorFlow, scikit-learn y matplotlib
Habitantes de Facatativá por las observaciones locales que validaron el modelo

📞 Contacto

Repositorio: https://github.com/brrojash/Proyecto-microclima-estacion-meteorologica
Issues: GitHub Issues

🔮 Posibles Extensiones Futuras
Aunque el proyecto está completado, se identificaron las siguientes oportunidades de expansión:

 Integración con APIs meteorológicas nacionales (IDEAM)
 Aplicación móvil para monitoreo remoto
 Expansión a otros microclimas de Cundinamarca
 Sistema de alertas tempranas
 Dashboard web en tiempo real
 Integración IoT con más sensores

📋 Cómo Citar Este Proyecto
Rojas, B., & Gutiérrez, N. (2024). Sistema de Predicción Meteorológica para Microclimas: 
Caso de Estudio Facatativá, Colombia. Proyecto de Grado, Universidad de Cundinamarca. 
GitHub: https://github.com/brrojash/Proyecto-microclima-estacion-meteorologica

Desarrollado con ❤️ para la comunidad meteorológica de Colombia
Versión: 1.0 (Final)
Fecha de Culminación: 2024
Estado: ✅ PROYECTO COMPLETADO
