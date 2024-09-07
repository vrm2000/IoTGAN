# IoTGAN: Generación y Análisis de Tráfico IoT

Este repositorio contiene los scripts y herramientas utilizados para generar y analizar tráfico IoT mediante el uso de redes adversarias generativas (GANs). Los principales objetivos de este proyecto incluyen:

- **Generación de Tráfico IoT Sintético**: Utilizando un modelo GAN entrenado para generar tráfico de red IoT que imita los patrones reales de tráfico.
- **Análisis y Procesamiento de Datos**: Scripts que procesan datasets reales de tráfico IoT y los preparan para el entrenamiento del modelo.
- **Envío de Paquetes Simulados**: Emisión de paquetes de red generados utilizando Scapy para simular el tráfico en un entorno de red real.
- **Entrenamiento de Modelos GAN**: Código para entrenar un modelo WGAN basado en TensorFlow y Keras, optimizado para datos de tráfico IoT.
- **Escalado de Datos**: Uso de técnicas de preprocesamiento para ajustar y escalar los datos de entrada para el entrenamiento del modelo GAN.

## Estructura del Repositorio

- **`IoTGAN.py`**: Script principal para la implementación y entrenamiento del modelo GAN.
- **`main.py`**: Script de control que maneja la generación de muestras, envío de paquetes y verificación del estado del modelo.
- **`utils.py`**: Funciones auxiliares para preprocesamiento de datos, escalado y generación de datasets.
- **`generator.py`**: Script que define la estructura del generador de la GAN
- **`discriminator.py`**: Script que define la estructura del discriminador de la GAN
- **`wgan_gp.py`**: Script que define funciones para convertir la GAN en WGAN-GP
- **`ACI-IoT-2023-Payload.csv`**: Fichero con el dataset que entrena el modelo. NEEDS TO BE DOWNLOADED AND PLACED WITH THE REST OF FILES.
- **`generated_samples.csv`**: Archivo de salida que almacena los paquetes generados por el modelo GAN.
- **`generator_model.keras`**: Modelo pre-entrenado de la GAN lista para ser usada.
- **`model_params.npy`**: Parámetros del modelo guardados para ser usados junto con el anterior fichero.
- **`requirements.txt`**: Fichero con todas las dependencias del proyecto

## Objetivos del Proyecto

Este proyecto ha sido desarrollado como prueba de concepto (POC) para el Trabajo Final de Máster de la Universidad de Málaga titulado "GENERACIÓN DE TRÁFICO SINTÉTICO EN DISPOSITIVOS DE INTERNET DE LAS COSAS MEDIANTE INTELIGENCIA ARTIFICIAL", tutorizado por Rodrigo Román Castro y desarrollado por Víctor Ramírez Mármol.

# Manual de Instalación y Uso

## Pasos para la instalación

1. **Clona el repositorio:**

    ```bash
   git clone https://github.com/tuusuario/turepositorio.git

2. **Descarga el dataset**
   Es importante descargar el dataset y posicionarlo en el mismo directorio que el resto de ficheros:
   -    https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?resource=download&select=ACI-IoT-2023-Payload.csv
4. **Instala las dependencias:**

    Ejecuta el siguiente comando para instalar todas las librerías necesarias desde el archivo requirements.txt:

   ```bash
        pip install -r requirements.txt
5. **¿Quieres generar un nuevo modelo?**

    Sí: Borra los archivos existentes del modelo entrenado antes de continuar.
      - model_params.npy
      - generator_model.keras
    No: Continua con el paso 4.

6. Ejecuta el script principal main.py:
    ```bash
      python main.py
 - Si se genera un nuevo modelo, el script IoTGAN.py será llamado automáticamente y se entrenará un nuevo modelo GAN.
 - Si ya existe un modelo entrenado, este será cargado y utilizado.
   
6. **Generación y envío de nuevos samples:**

Nuevos samples serán generados y enviados como paquetes usando scapy. Los paquetes pueden ser capturados con herramientas como Wireshark. Además, los samples generados se irán añadiendo al archivo generate_samples.csv.
