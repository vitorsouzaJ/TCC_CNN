import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Definir os caminhos das pastas de treinamento, validação e teste
train_dir = './data_set_orquideas/train'
val_dir = './data_set_orquideas/validation'
test_dir = './data_set_orquideas/test'


#train_dir = './data_test/train'
#val_dir = './data_test/validation'
#test_dir = './data_test/test'

# Definir o tamanho das imagens de entrada
input_shape = (224, 224, 3)  # Tamanho utilizado pela VGGNet

# Definir o número de classes
num_classes = 4

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    rotation_range=90,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True

)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    classes=['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum']
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    classes=['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum']
)

model = Sequential([
    Conv2D(64, (11, 11), activation='relu', input_shape=input_shape),
    MaxPooling2D((5, 5)),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')

])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Configurar o callback ModelCheckpoint para salvar o modelo em formato .keras
checkpoint_callback_keras = ModelCheckpoint(
    filepath='./modelosNeurais/modelos_VGGNet.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Configurar o callback ModelCheckpoint para salvar o modelo em formato .h5
checkpoint_callback_h5 = ModelCheckpoint(
    filepath='./modelosNeurais/modelos_VGGNet.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Treinar o modelo
history = model.fit(
    train_generator, 
    epochs=30, 
    validation_data=validation_generator,
    callbacks=[checkpoint_callback_keras, checkpoint_callback_h5]
)

# Avaliar o modelo no conjunto de teste
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=1,  # Alterado para batch_size=1 para obter previsões por amostra
    class_mode='categorical',
    shuffle=False,  # Desativar o embaralhamento para garantir correspondência entre previsões e rótulos
    classes=['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum']
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)

# Fazer previsões nos dados de teste
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Obter rótulos reais dos dados de teste
true_labels = test_generator.classes

## Calcular métricas
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print('=============================')
print('Acuracia:', accuracy)
print('Precisao:', precision)
print('Recall:', recall)
print('F1-score:', f1)
print()
print()


accuracy_per_class = []
total_samples_per_class = np.bincount(true_labels)
for i, class_name in enumerate(['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum']):
    correct_predictions = np.logical_and(predicted_labels == true_labels, true_labels == i)
    class_accuracy = np.sum(correct_predictions) / total_samples_per_class[i]
    accuracy_per_class.append(class_accuracy)

precision_per_class = precision_score(true_labels, predicted_labels, labels=[0, 1, 2, 3], average=None, zero_division=1)
recall_per_class = recall_score(true_labels, predicted_labels, labels=[0, 1, 2, 3], average=None, zero_division=1)
f1_per_class = f1_score(true_labels, predicted_labels, labels=[0, 1, 2, 3], average=None, zero_division=1)

for i, class_name in enumerate(['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum']):
    print('=============================')
    print('Classe:', class_name)
    print('Acuracia:', accuracy_per_class[i])
    print('Precisao:', precision_per_class[i])
    print('Recall:', recall_per_class[i])
    print('F1-score:', f1_per_class[i])
    print()

# Obter histórico de treinamento
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Criar gráfico da precisão
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy - VGGNet')
plt.legend()
plt.show()

# Criar gráfico da perda
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - VGGNet')
plt.legend()
plt.show()



# Obtendo a matriz de confusão
confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)

# Definindo os nomes das classes
class_names = ['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum']

# Criando um mapa de calor da matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names,
            fmt='d', cbar=False)

# Adicionando as porcentagens à matriz de confusão
total_samples_per_class = tf.reduce_sum(confusion_matrix, axis=1)
error_percentage = 1.0 - tf.linalg.diag_part(confusion_matrix) / total_samples_per_class
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        if i != j:
            plt.text(j + 0.5, i + 0.5, f'{confusion_matrix[i][j]}', ha='center', va='center', color='white')

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - VGGNet')
plt.show()

# Plotar gráfico do recall por classe
plt.bar(['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum'], recall_per_class)
plt.xlabel('Classe')
plt.ylabel('Recall')
plt.title('Recall por Classe  - VGGNet')
plt.show()

# Plotar gráfico do F1-score por classe
plt.bar(['Brassia', 'Zygopetalum', 'Vanda', 'Angraecum'], f1_per_class)
plt.xlabel('Classe')
plt.ylabel('F1-score')
plt.title('F1-score por Classe - VGGNet')
plt.show()



# Extrair os pesos da rede neural
weights = []
biases = []
for layer in model.layers:
    if layer.get_weights():
        layer_weights = layer.get_weights()[0]
        layer_biases = layer.get_weights()[1]
        weights.append(layer_weights)
        biases.append(layer_biases)

# Salvar os pesos e biases em um arquivo usando pickle
data = {'weights': weights, 'biases': biases}
with open('./modelosNeurais/modelo_VGGNet.pkl', 'wb') as file:
    pickle.dump(data, file)