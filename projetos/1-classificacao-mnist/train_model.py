import tensorflow as tf
import os
#from tensorflow import keras
#from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Carregamento do dataset MNIST via TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pré-processamento dos dados
# Adiciona a dimensão do canal:
# (28, 28) -> (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Converte os valores para float32 e normaliza entre 0 e 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Imagens de treino:", x_train.shape[0])
print("Imagens de teste:", x_test.shape[0])

# Construção da CNN
#    blocos convolucionais:
model = Sequential([
    # Bloco convolucional 1
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(28, 28, 1)
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Bloco convolucional 2
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Bloco convolucional 3
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Conversão dos mapas de características em vetor
    Flatten(),

    # Camada densa
    Dense(128, activation="relu"),

    # Dropout antes da camada de saída
    Dropout(0.5),

    # Camada de saída para as 10 classes do MNIST
    Dense(10, activation="softmax")
])

# Compilação do modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early Stopping baseado na perda de validação
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Treinamento do modelo
history = model.fit(
    x_train,
    y_train,
    validation_split=0.20,
    epochs=15,
    batch_size=32,
    callbacks=[early_stopping],
    shuffle=True
)

# Exibição da acurácia final de validação
val_loss = history.history["val_loss"][-1]
val_accuracy = history.history["val_accuracy"][-1]

print("\nResultados finais de validação")
print(f"Perda de validação: {val_loss:.4f}")
print(f"Acurácia de validação: {val_accuracy:.4f}")
print(
    f"Acurácia de validação em porcentagem: "
    f"{val_accuracy * 100:.2f}%"
)

# Salvamento do modelo treinado
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.h5")
model.save(model_path)