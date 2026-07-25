# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Rodrigo da Silva Santos

### 1️⃣ Resumo da Arquitetura do Modelo

Descreva, em palavras, a arquitetura da CNN implementada em `train_model.py` (número de blocos convolucionais, uso de batch normalization/dropout, estratégia de validação/early stopping).

 CNN implementada possui três blocos convolucionais. Uma camada Conv2D, responsável por extrair 
características das imagens, uma camada BatchNormalization, que normaliza as ativações e ajuda a 
tornar o treinamento mais estável uma camada MaxPooling2D, que reduz as dimensões espaciais dos mapas 
de características. No primeiro bloco, a convolução utiliza 32 filtros. No segundo, são usados 64 
filtros. No terceiro, são usados 128 filtros. Todas as convoluções possuem kernels de tamanho 3 × 3, 
ativação ReLU e preenchimento same.
Após os três blocos convolucionais, a camada Flatten transforma os mapas de características em um 
vetor.Esse vetor é enviado para uma camada densa com 128 neurônios e ativação ReLU.
Em seguida, é aplicada uma camada Dropout(0.5), que desativa aleatoriamente 50% dos neurônios durante 
o treinamento. Essa técnica ajuda a reduzir o sobreajuste.
A validação é realizada por meio do parâmetro validation_split=0.20. Dessa forma, 20% das imagens 
originalmente fornecidas para treinamento são separadas automaticamente para compor o conjunto de 
validação, enquanto os 80% restantes são utilizados efetivamente no treinamento.

### 2️⃣ Bibliotecas Utilizadas

Liste as principais bibliotecas utilizadas, preferencialmente com suas versões.

. Tensorflow
. Numpy
. os

### 3️⃣ Técnica de Otimização do Modelo

Explique qual técnica foi utilizada para otimizar o modelo em `optimize_model.py`.

A técnica de otimização utilizada foi a Dynamic Range Quantization, técnica especialmente utilizada 
no Tensorflow Lite para reduzir o tamanho do modelo e acelerar a inferência, pricipalmente em 
dispositivos com recursos limitados.

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.

. Acurácia de validação obtida: 0.9920 ou 99.20%
. model.h5: 2.9M
. model.tflite: 251k

### 5️⃣ Comentários Adicionais (Opcional)

Dificuldades encontradas, decisões técnicas importantes, limitações do modelo, aprendizados durante o desafio.

### 6️⃣ Exemplo de Inferência

Cole a saída do terminal ao rodar `run_inference.py` (predito vs. real para as 5+ amostras), e comente brevemente se houve algum caso interessante (acerto ou erro) entre as amostras testadas.

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Nas cinco amostras testadas, o modelo classificou corretamente todos os dígitos, houve concordância 
entre a classe predita e a classe real em 100% dos casos apresentados.