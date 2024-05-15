import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy 
x_train = np.load('./results/x_train.npy')
x_valid = np.load('./results/x_valid.npy')
y_train = np.load('./results/y_train.npy')
y_valid = np.load('./results/y_valid.npy')
pred_img= np.load('./results/pred_img.npy')
result=x_valid[:, :, :, 6:]
x_train =  x_train[:,:,:,0:6]
x_valid = x_valid[:,:,:,0:6]
y_train = y_train[:,:,:,0:6]
y_valid = y_valid[:,:,:,0:6]
print(pred_img.shape)
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

precision.update_state(y_valid, pred_img)
recall.update_state(y_valid, pred_img)
accuracy.update_state(y_valid, pred_img)
precision_result = precision.result().numpy()
recall_result = recall.result().numpy()
accuracy_result = accuracy.result().numpy()

# Calcula F1 Score
f1_score = 2 * (precision_result * recall_result) / (precision_result + recall_result + 1e-7)  # Evita división por cero

print(f"Accuracy: {accuracy_result}, Precision: {precision_result}, Recall: {recall_result}, F1 Score: {f1_score}")

img=0
plt.figure()
plt.imshow(pred_img[img, :, :, 0],cmap='gray')
plt.title("Predictions")
plt.axis('off')
plt.figure()
plt.imshow(y_valid[img, :, :, 0],cmap='gray')
plt.title("Label")
plt.axis('off')
result=result[img, :, :, :]
plt.figure()
plt.imshow(result)
plt.title('Original Image')
plt.axis('off')
plt.show()
result=result/ result.max()
result[:,:,0] =result[:,:,0]*pred_img[img, :, :, 0]
result[:,:,1] =result[:,:,1]*pred_img[img, :, :, 0]
result[:,:,2] =result[:,:,2]*pred_img[img, :, :, 0]

plt.figure()
plt.imshow(result)
plt.title('Segmented Image')
plt.axis('off')
plt.show()