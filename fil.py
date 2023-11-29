import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt




data = pd.read_csv("student-mat.csv", sep=";")


features = ["G1", "G2", "G3", "studytime", "failures", "absences"]
data = data[features]

predict_column = "G3"


X = data.drop(columns=[predict_column])
y = data[predict_column]


split_ratio = 0.9
split_index = int(len(data) * split_ratio)

X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]


model = Sequential([
    Dense(units=1, input_dim=X_train.shape[1], activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, epochs=50, verbose=0)


loss = model.evaluate(X_test, y_test)
print(f'Model Loss on Test Set: {loss:.4f}')


plt.plot(history.history['loss'])
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


predictions = model.predict(X_test).flatten()


for i in range(len(predictions)):
    print(f"Input {i + 1}:")
    print(f"Predicted: {predictions[i]}")
    print(f"Actual value: {y_test.iloc[i]}")


window = tk.Tk()
window.title("Final Grade Prediction with Linear Regression")
window.geometry("800x600")


def choose_file():
    file_path = filedialog.askopenfilename()
    print(f'Selected file: {file_path}')


file_button = tk.Button(window, text="Choose File", command=choose_file)
file_button.pack()

plt.figure()
plt.plot(history.history['loss'])
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

window.mainloop()
