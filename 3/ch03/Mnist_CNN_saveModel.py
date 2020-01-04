import numpy as np
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense

#建立訓練資料和測試資料，包括訓練特徵集、訓練標籤和測試特徵集、測試標籤	
(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()  

#將 Features 特徵值換為 60000*28*28*1 的 4 維矩陣
train_feature_vector =train_feature.reshape(len(train_feature), 800,400,1).astype('float32')
test_feature_vector = test_feature.reshape(len( test_feature), 800,400,1).astype('float32')

#Features 特徵值標準化
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

#label 轉換為 One-Hot Encoding 編碼
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

#建立模型
model = Sequential()
#建立卷積層1
model.add(Conv2D(filters=10, 
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(800,400,1), 
                 activation='relu'))

#建立池化層1
model.add(MaxPooling2D(pool_size=(2, 2))) #(10,14,14)

#建立卷積層2
model.add(Conv2D(filters=20, 
                 kernel_size=(5,5),  
                 padding='same',
                 activation='relu'))

#建立池化層2
model.add(MaxPooling2D(pool_size=(2, 2))) #(20,7,7)

# Dropout層防止過度擬合，斷開比例:0.2
model.add(Dropout(0.2))

#建立平坦層：20*7*7=980 個神經元
model.add(Flatten()) 

#建立隱藏層
model.add(Dense(units=256, activation='relu'))

#建立輸出層
model.add(Dense(units=10,activation='softmax'))

# 這些訓練會累積，準確會愈來愈高
try:
    model.load_weights("Mnist_cnn_model.weight")
    print("載入模型參數成功，繼續訓練模型!")
except :    
    print("載入模型失敗，開始訓練一個新模型!")

#定義訓練方式
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

#以(train_feature_normalize,train_label_onehot)資料訓練，
#訓練資料保留 20% 作驗證,訓練10次、每批次讀取200筆資料，顯示簡易訓練過程
train_history =model.fit(x=train_feature_normalize,
                         y=train_label_onehot,validation_split=0.2, 
                         epochs=10, batch_size=200,verbose=2)
#評估準確率
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print('\n準確率=',scores[1])
    
# 儲存模型
model.save('Mnist_cnn_model.h5')
print("\nMnist_cnn_model.h5 模型儲存完畢!")
model.save_weights("Mnist_cnn_model.weight")
print("Mnist_cnn_model.weight 模型參數儲存完畢!")

del model       