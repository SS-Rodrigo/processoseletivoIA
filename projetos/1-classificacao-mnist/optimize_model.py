import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Caminho da pasta onde este arquivo está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos dos arquivos
model_h5_path = os.path.join(script_dir, "model.h5")
model_tflite_path = os.path.join(script_dir, "model.tflite")


# 1. Carregar o modelo treinado model.h5
print('Carregando o modelo "model.h5"...')

model = tf.keras.models.load_model(
    model_h5_path,
    compile=False
)

print("Modelo carregado com sucesso.")


# 2. Criar o conversor para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)


# 3. Aplicar Dynamic Range Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Convertendo e otimizando o modelo...")


# Realizar a conversão
model_tflite = converter.convert()


# 4. Salvar o modelo convertido como model.tflite
with open(model_tflite_path, "wb") as arquivo:
    arquivo.write(model_tflite)
