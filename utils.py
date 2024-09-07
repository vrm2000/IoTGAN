import pandas as pd

import pandas as pd

def preprocess_data(file_path, process_chunk):
    chunk_size = 10000
    first_chunk = True

    for chunk in pd.read_csv(file_path, delimiter=',', chunksize=chunk_size):
        # Llama a process_chunk solo con un argumento (sin scaler)
        processed_chunk = process_chunk(chunk)
        
        if first_chunk:
            processed_chunk.to_csv("prepared_data.csv", index=False, mode='w', header=True)
            first_chunk = False
        else:
            processed_chunk.to_csv("prepared_data.csv", index=False, mode='a', header=False)

    prepared_data = pd.read_csv("prepared_data.csv")
    return prepared_data.values


def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)
