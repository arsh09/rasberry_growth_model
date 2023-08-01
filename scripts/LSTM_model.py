# %%
# imports 
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
import keras.backend as k_backend

import cv2
import imutils
import time

# %%
# data files 
base_path = "./../data/"
files = [os.path.join("{}{}".format(base_path,i)) for i in os.listdir(base_path) if os.path.exists("{}{}".format(base_path,i))]

# %%
 # get single sample and label from dataframe
def get_single_sample(df_sub):
    x1, x2, y1, y2 = df_sub["bbox_x1"], df_sub["bbox_x2"], df_sub["bbox_y1"], df_sub["bbox_y2"]
    xc, yc = (x1 + 2)/2, (y1+y2)/2
    width, height = x2 - x1 , y2 - y1
    r, g, b = df_sub["median_r"], df_sub["median_g"], df_sub["median_b"]
    h, s, v = df_sub["median_h"], df_sub["median_s"], df_sub["median_v"]
    stype = df_sub['type']
    sample = np.vstack( ( x1, x2, y1, y2, r, g, b, h, s, v, stype ) ).T 

    return sample


def get_single_label(df_label):
    x1, x2, y1, y2 = df_label["bbox_x1"], df_label["bbox_x2"], df_label["bbox_y1"], df_label["bbox_y2"]
    xc, yc = (x1 + 2)/2, (y1+y2)/2
    width, height = x2 - x1 , y2 - y1
    r, g, b = df_label["mean_r"], df_label["mean_g"], df_label["mean_b"]
    stype = df_label['type']
    label = np.vstack( ( x1, x2, y1, y2  ) ).T

    return label

# %%
# preprocessing

look_back = 4 # input time length
look_forward = 1 # output time length

x_dataset = []
y_dataset = []

for f in files: 
    
    df = pd.read_csv(f, sep='\t')

    if len(df) > look_back + look_forward: 

        for i in range( len(df) - look_back - look_forward ): 

            df_sub = df.iloc[i: i+look_back]
            df_label = df.iloc[i+look_back: i+look_back+look_forward]

            # train set
            sample = get_single_sample(df_sub)
            x_dataset.append( sample )

            # test set
            label = get_single_label(df_label)
            y_dataset.append(label)
            
# prepared dataset
x_dataset = np.array(x_dataset)
y_dataset = np.array(y_dataset)

print ("Total Data: ", x_dataset.shape, y_dataset.shape)

# %%
# split train, test and validation sets 
n = x_dataset.shape[0] 

X_train, y_train = x_dataset[0 : int(0.7*n)], y_dataset[0 : int(0.7 * n)]
X_test, y_test = x_dataset[int(0.7*n) : int(0.9*n)], y_dataset[int(0.7 * n) : int(0.9*n)]
X_val, y_val = x_dataset[int(0.9*n) : ], y_dataset[int(0.9*n) : ]


print ("Train Dataset: ", X_train.shape, y_train.shape)
print ("Test Dataset: ", X_test.shape, y_test.shape)
print ("Validation Dataset: ", X_val.shape, y_val.shape)
print ("-"*50)

print ("Sample Train Dataset: ", X_train[0].shape, y_train[0].shape)
print ("Sample Test Dataset: ", X_test[0].shape, y_test[0].shape)
print ("Sample Validation Dataset: ", X_val[0].shape, y_val[0].shape)
print ("-"*50)

print ("Train Max/Min | Mean | Std.: ", np.max(X_train), np.min(X_train), np.mean(X_train), np.std(X_train),  )
print ("Test Max/Min | Mean | Std.: ", np.max(X_test), np.min(X_test), np.mean(X_test), np.std(X_test) )
print ("Val Max/Min | Mean | Std.: ", np.max(X_val), np.min(X_val), np.mean(X_val), np.std(X_val) )


# %%
# RMSE custom loss - not really used.
def root_mean_squared_error(y_true, y_pred):
    return k_backend.sqrt(k_backend.mean(k_backend.square(y_pred - y_true)))

# %%
# model 
n_features = X_train.shape[2]
n_neurons = 200
n_epochs = 250
n_out_features = y_train.shape[-1]

model = Sequential()
model.add( LSTM(n_neurons, input_shape=(look_back, n_features), return_sequences = True, name="lstm-1-layer")) 
model.add( LSTM(int(n_neurons/2), return_sequences = False  , activation = "relu", name = "lstm-2-layer") )
model.add( Dense(look_forward*n_out_features, activation = "sigmoid" , name = "dense-layer") )
model.add( Reshape([look_forward, n_out_features], name = "reshape-layer") )

opt = keras.optimizers.Adam(learning_rate=0.1)

# model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['accuracy', 'mse', 'mae', root_mean_squared_error])
# model.compile(loss="mae", optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
model.compile(loss="mse", optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
model.summary()

# %%
# training 
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=50)

history = model.fit(
    x = X_train, y = y_train,
    epochs=n_epochs, 
    verbose=1,
    validation_data=(X_val, y_val),
    # callbacks=[early_stopping, reduce_lr],
    shuffle = False
)

# %%
print (history.history.keys())
# plot model training history
plt.figure(figsize=(15,7))
plt.subplot(2,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title("Loss (MSE)")
plt.legend()

plt.subplot(2,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.legend()

plt.subplot(2,2,3)
plt.plot(history.history['mse'], label='train')
plt.plot(history.history['val_mse'], label='val')
plt.title("MSE")
plt.legend()

plt.subplot(2,2,4)
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='val')
plt.title("MAE")
plt.legend()

plt.savefig("./rnn_{}}_lookback_{}_lookahead_{}_features_in_{}_features_out.png".format(look_back, look_forward, n_features, n_out_features))

plt.show()

# %%
# predictions
y_predictions  = model.predict(X_test)
print (y_predictions.shape, y_test.shape)

# %%
# simple check
print ("Predicted: \n", y_predictions[0])
print ("-"*50)
print ("Real: \n", y_test[0])

# %%
#evaluation metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef( np.squeeze(forecast), np.squeeze(actual) )[0,1]   # corr
    
    return({'mape':mape*100, 'me':me, 'mae': mae,  'rmse':rmse, 
            'corr':corr})


for i in range(n_out_features):
    out = (forecast_accuracy( y_predictions[:,:,i], y_test[:,:,i] ))
    print ("Feature : {}".format(i+1))
    print ("MAPE:\t", out["mape"], " %")
    print ("ME:\t", out["me"])
    print ("MAE:\t", out["mae"])
    print ("RMSE:\t", out["rmse"])
    print ("CORR:\t", out["corr"])
    print ("-----------")

# %%
# test on random sets.
random_n = np.random.randint( X_test.shape[0]  )

test_y, label_y, predict_y = X_test[random_n], y_test[random_n], y_predictions[random_n]

test_x = np.linspace( 0, look_back , num = look_back )
label_x = np.arange( look_back + 1, look_forward + look_back + 1 , 1 )


plt.figure(figsize = (42, 6))
for i in range(n_out_features):
    plt.subplot(1, n_features+1, i+1)
    plt.plot(test_x, test_y[:, i] , '-r', label = "input" ) 
    plt.plot(label_x, label_y[:, i] , '-xb', label = "actual" )
    plt.plot(label_x, predict_y[:, i]  , '-og', label = "predicted" )
    plt.title("Feature: {}".format(i+1))
    plt.ylim([np.min(test_y[:,i]) - 0.05, np.max(test_y[:,i]) + 0.05 ])
    plt.legend()

plt.show()

# %%
# This block takes a random file and 
# predicts the future bbox (x1, x2, y1, y2) 

# model.save ("./../../models/berry_lstm_in_feature_11_out_feature_4_look_back_4_look_ahead_1_n_epochs_250.h5")
# model = keras.models.load_model("./../../models/berry_lstm_in_feature_11_out_feature_4_look_back_4_look_ahead_1_n_epochs_250.h5")

# test on random sets.
random_n =  np.random.randint( len(files) )
df_sub = pd.read_csv(files[random_n], sep='\t')

# train set
sample = get_single_sample( df_sub )
sample_real = get_single_sample (df_sub)

predicted_values = []

for i in range(look_back):
    predicted_values.append([0,0,0,0])

for i in range(sample.shape[0] - look_back ):
    
    seq = sample[i : i + look_back]  
    y_predict = model.predict( np.expand_dims( seq , axis=0), verbose=0)  
    predicted_values.append( y_predict.squeeze().tolist() )

    # For teacher forcing. Otherwise comment this.
    sample[i+look_back, 0:4] = y_predict.squeeze().tolist()



# this plots predictions
predicted_values = np.array( predicted_values )
feature_names = ["x1", "x2", "y1", "y2"]

plt.figure(12, figsize=(16,6))
for i in range(n_out_features):

    plt.subplot(2,2,i+1)
    plt.plot( sample_real[:, i] , 'rx--', label="Real {}".format(feature_names[i]))

    # For teacher forcing. otherwise comment this.
    plt.plot( sample[:, i] , 'go--', label="Pred {}".format(feature_names[i]))

    # For normal prediction, 
    # plt.plot( predicted_values[:, i] , 'go--', label="Pred {}".format(feature_names[i]))
    plt.legend()

    # this gets used in the next block to visualize the predicted bbox.   
    df_sub['predicted_{}'.format(feature_names[i])] = predicted_values[:,i]


# this plots features 
other_features = [ 
    ['mean_r', 'mean_g','mean_b', 'type'], ['median_r', 'median_g','median_b', 'type'],
    ['mean_h', 'mean_s','mean_v', 'type'], ['median_h', 'median_s','median_v', 'type'],
    ['type'], [ 'bbox_y1', 'bbox_y2','type','type',] 
]

colors = ['rx--', 'gx--', 'bx--', 'k']

plt.figure(13, figsize=(16,6))

for i in range( len(other_features) ):

    if len(other_features) % 2 == 0:
        plt.subplot( int (len(other_features)/2) ,2 , i+1 )
    else:
        plt.subplot( len(other_features),1, i+1 )

    for count, feature in enumerate( other_features[i] ):
        plt.plot(  df_sub[ feature] , colors[count], label=feature )
        plt.legend()
        plt.ylim([-0.1,1.1])


# %%
# Correct the path 'img_folder'

img_folder = "/path/to/raw/images"

for i in range(len(df_sub)):
    row = df_sub.iloc[i]

    img_path = os.path.join( img_folder, "{}{}".format(row['filename'], '.png') )
    if os.path.exists(img_path):

        img = cv2.imread(img_path)
        img = imutils.resize(img, width = 640)
        h, w = img.shape[:2]

        # plot label
        x1, x2, y1, y2 = int(row['bbox_x1'] * w), int(row['bbox_x2'] * w), int (row['bbox_y1'] * h), int(row['bbox_y2'] * h)
        cv2.rectangle( img, (x1,y1), (x2, y2), (0, 0, 255), 2 )

        # plot label
        px1, px2, py1, py2 = int(row['predicted_x1'] * w), int(row['predicted_x2'] * w), int (row['predicted_y1'] * h), int(row['predicted_y2'] * h)
        cv2.rectangle( img, (px1,py1), (px2, py2), (0, 255, 0), 2 )
        
        cv2.imshow('image', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)


cv2.destroyAllWindows()



