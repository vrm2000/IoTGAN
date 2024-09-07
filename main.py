import os, sys, io
import subprocess
import random
import psutil
import string
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
from colorama import init, Fore, Style
from scapy.all import IP, TCP, UDP, send

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Ruta del intérprete de Python en el entorno virtual
python_executable = sys.executable

# Ruta del modelo guardado
generator_model_path = 'generator_model.keras'
params_file_path = 'model_params.npy'

# Inicialización para vaciar el archivo al inicio
file_path_packets = 'generated_samples.csv'
def initialize_file(file_path=file_path_packets):
    # Comprobar si el archivo ya existe antes de vaciarlo
    if os.path.exists(file_path):
        print(f"Archivo {file_path} ya existe. Será sobrescrito.")
    else:
        print(f"Archivo {file_path} no existe. Creando uno nuevo.")

    # Abrir el archivo en modo escritura para vaciarlo
    with open(file_path, 'w') as file:
        # Escribir el encabezado
        file.write('srcip,sport,dstip,dsport,protocol,stime,total_len,payload\n')

# Llamada a la función para inicializar el archivo
initialize_file('generated_samples.csv')

# Función para verificar si el modelo y los parámetros existen y entrenar si no existen
def check_and_train_model():
    if not os.path.exists(generator_model_path) or not os.path.exists(params_file_path):
        print(Fore.RED + f"El modelo o los parámetros no existen. Entrenando un nuevo modelo..." + Style.RESET_ALL)

        # Ejecutar el script de entrenamiento utilizando el intérprete de Python del entorno virtual
        result = subprocess.run([python_executable, 'IoTGAN.py'], capture_output=True, text=True)

        # Mostrar la salida del script de entrenamiento (opcional)
        print(result.stdout)

        # Comprobar si hubo algún error
        if result.returncode != 0:
            print(f"Error al entrenar el modelo: {result.stderr}")
            return None
    else:
        print(Fore.YELLOW + f"El modelo {generator_model_path} y el archivo de parámetros {params_file_path} ya existen." + Style.RESET_ALL)

    # Cargar el modelo entrenado
    generator = load_model(generator_model_path)
    print(Fore.GREEN + f"Modelo cargado correctamente desde {generator_model_path}" + Style.RESET_ALL)

    # Cargar scaler y latent_dim desde el archivo de parámetros
    model_params = np.load(params_file_path, allow_pickle=True).item()
    scaler = model_params['scaler']
    latent_dim = model_params['latent_dim']
    
    return generator, scaler, latent_dim

# Función para generar un sample, procesarlo y enviarlo con scapy
def generate_and_send_sample(generator, latent_dim, real_data, scaler):
    # Generar un único sample
    noise = np.random.normal(0, 1, (1, latent_dim)).astype(np.float32)
    generated_data = generator.predict(noise)

    # Convertir el array generado en un DataFrame temporal
    generated_df = pd.DataFrame(generated_data, columns=['total_len', 'protocol_TCP', 'protocol_UDP', 'protocol_Other', 'payload'])

    # Desescalar total_len
    generated_df['total_len'] = scaler.inverse_transform(generated_df[['total_len']])

    # Asegurarse de que los valores de 'total_len' sean positivos
    generated_df['total_len'] = np.abs(generated_df['total_len'])
    generated_df['total_len'] = np.round(generated_df['total_len']).astype(int)

    # Relacionar 'total_len' con el tamaño del 'payload'
    def convert_payload(total_len):
        payload_size = max(1, int(np.abs(total_len / 10)))
        payload = ''.join(random.choices(string.hexdigits, k=payload_size))
        return payload.lower()

    # Asignar el payload generado en función de 'total_len'
    generated_df['payload'] = generated_df['total_len'].apply(convert_payload)

    # Seleccionar srcip, dstip, sport, dsport aleatoriamente de los datos reales
    srcip = real_data['srcip'].sample(n=1, replace=True).values[0]
    dstip = real_data['dstip'].sample(n=1, replace=True).values[0]
    sport = real_data['sport'].sample(n=1, replace=True).values[0]
    dsport = real_data['dsport'].sample(n=1, replace=True).values[0]

    current_time = int(time.time())

    # Seleccionar el protocolo basado en los valores generados
    protocol_columns = ['protocol_TCP', 'protocol_UDP', 'protocol_Other']
    max_protocol = generated_df[protocol_columns].idxmax(axis=1).values[0]
    protocol = {
        'protocol_TCP': 'TCP',
        'protocol_UDP': 'UDP',
        'protocol_Other': 'Other'
    }.get(max_protocol, 'TCP')

    # Obtener el total_len y payload
    total_len = generated_df['total_len'].values[0]
    payload = generated_df['payload'].values[0]


    # Guardar el paquete en el archivo (append)
    packet_data = {
        'srcip': srcip,
        'sport': sport,
        'dstip': dstip,
        'dsport': dsport,
        'protocol': protocol,
        'stime': current_time,
        'total_len': total_len,
        'payload': payload
    }

    # Convertir a DataFrame para hacer append fácilmente
    packet_df = pd.DataFrame([packet_data])

    # Hacer append en el archivo CSV
    packet_df.to_csv(file_path_packets, mode='a', header=False, index=False)

    # Procesar y enviar el paquete usando scapy
    send_packet(srcip, dstip, sport, dsport, protocol, payload)

# Función para enviar un paquete generado
def send_packet(src_ip, dst_ip, sport, dport, protocol, payload):
    if protocol == "TCP":
        pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport) / payload
    elif protocol == "UDP":
        pkt = IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / payload
    else:
        print(f"Protocolo {protocol} no soportado.")
        return
    
    # Enviar el paquete
    print(Fore.LIGHTMAGENTA_EX + f"Enviando paquete: {protocol} de {src_ip}:{sport} a {dst_ip}:{dport} con payload: {payload}" + Style.RESET_ALL)
    send(pkt)

# Comprobar y cargar el modelo y los parámetros
result = check_and_train_model()
if result:
    generator, scaler, latent_dim = result

    # Usar el modelo si fue cargado correctamente
    file_path = r".\\ACI-IoT-2023-Payload.csv"
    real_data = pd.read_csv(file_path)

    while True:
        generate_and_send_sample(generator, latent_dim, real_data, scaler)
        time.sleep(10)  # Espera de 10 segundos entre envíos
