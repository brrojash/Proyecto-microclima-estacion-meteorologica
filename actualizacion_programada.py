import os
import sys
import schedule
import time
from datetime import datetime

# Añadir directorio actual al path para importar módulos
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Importar funciones necesarias
from meteo_main import integrar_datasets
from predictor_model import PrediccionMeteo

def tarea_actualizacion():
    try:
        print(f"Iniciando actualización programada: {datetime.now()}")
        
        # 1. Obtener datos más recientes
        ruta_nuevos = "datos_estacion_ultimo_dia.csv"
        
        # 2. Cargar primero para analizar nuevas categorías
        predictor = PrediccionMeteo()
        predictor.cargar_modelo_guardado()
        
        # 3. Cargar nuevos datos para encontrar categorías
        dataset_nuevos = predictor.cargar_datos(ruta_nuevos)
        
        # 4. Actualizar categorías antes de integrar
        nuevas_categorias = set()
        for _, row in dataset_nuevos.iterrows():
            categoria = predictor.categorizar_clima(row)
            nuevas_categorias.add(categoria)
        
        # 5. Agregar nuevas categorías al encoder
        if hasattr(predictor, 'categorias') and predictor.categorias is not None:
            todas_categorias = set(predictor.categorias).union(nuevas_categorias)
            predictor.categorias = sorted(list(todas_categorias))
            predictor.num_categorias = len(predictor.categorias)
            predictor.label_encoder.fit(predictor.categorias)
            print(f"Categorías actualizadas: {len(predictor.categorias)}")
        
        # 6. Integrar con datos históricos
        dataset_completo = integrar_datasets(
            "dataset_historico.csv", 
            ruta_nuevos,
            "dataset_completo_actualizado.csv"
        )
        
        # 7. Actualizar modelo con nuevos datos
        predictor.actualizar_modelo_con_nuevos_datos(
            "dataset_completo_actualizado.csv"
        )
        
        print(f"Actualización completada: {datetime.now()}")
        
    except Exception as e:
        print(f"Error en actualización programada: {str(e)}")

def iniciar_actualizacion_diaria():
    """Inicia el sistema de actualización diaria"""
    # Programar actualización diaria a las 02:00 AM
    schedule.every().day.at("02:00").do(tarea_actualizacion)
    
    # También permitir actualización manual
    print("Sistema de actualización iniciado. Presiona 'q' para salir.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verificar cada minuto

if __name__ == "__main__":
    iniciar_actualizacion_diaria()
######## By: Bryan Rojas and Nathalia Gutierrez ########
# 2024-01-01
# Actualización programada para el modelo de microclima