if __name__ == '__main__':
    import torch
    from ultralytics import YOLO

    # Configurar el modelo
    model = YOLO('yolov8n.yaml')

    # Iniciar el entrenamiento con early stopping (patience=20)
    results = model.train(
        data='data.yaml',   # Ruta a tu archivo de datos
        epochs=1000,         # Número de épocas de entrenamiento
        patience=50      # Paciencia para early stopping
    )
