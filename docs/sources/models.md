## Sequential

Linear stack of layers.

```python
model = keras.models.Sequential()
```
- __Methods__:
    - __add(layer)__: Add a layer to the model.
    - __compile(optimizer, loss, class_mode="categorical")__: 
        - __Arguments__: 
            - __optimizer__: str (name of optimizer) or optimizer object. See [optimizers](optimizers.md).
            - __loss__: str (name of objective function) or objective function. See [objectives](objectives.md).
            - __class_mode__: one of "categorical", "binary". This is only used for computing classification accuracy or using the predict_classes method. 
    - __fit(X, y, batch_size=128, nb_epoch=100, verbose=1, validation_split=0., validation_data=None, shuffle=True, show_accuracy=False)__: Train a model for a fixed number of epochs.
        - __Arguments__: 
            - __X__: data.
            - __y__: labels.
            - __batch_size__: int. Number of samples per gradient update.
            - __nb_epoch__: int. 
            - __verbose__: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
            - __validation_split__: float (0. < x < 1). Fraction of the data to use as held-out validation data.
            - __validation_data__: tuple (X, y) to be used as held-out validation data. Will override validation_split.
            - __shuffle__: boolean. Whether to shuffle the samples at each epoch.
            - __show_accuracy__: boolean. Whether to display class accuracy in the logs to stdout at each epoch.
    - __evaluate(X, y, batch_size=128, show_accuracy=False, verbose=1)__: Show performance of the model over some validation data.
        - __Arguments__: Same meaning as fit method above. verbose is used as a binary flag (progress bar or nothing).
    - __predict_proba(X, batch_size=128, verbose=1)__: Return an array of predictions for some test data.
        - __Arguments__: Same meaning as fit method above.
    - __predict_classes(X, batch_size=128, verbose=1)__: Return an array of class predictions for some test data.
        - __Arguments__: Same meaning as fit method above. verbose is used as a binary flag (progress bar or nothing).
    - __train__(X, y, accuracy=False)__: Single gradient update on one batch. if accuracy==False, return tuple (loss_on_batch, accuracy_on_batch). Else, return loss_on_batch.
    - __test__(X, y, accuracy=False)__: Single performance evaluation on one batch. if accuracy==False, return tuple (loss_on_batch, accuracy_on_batch). Else, return loss_on_batch.


__Examples__:

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, 2, init='uniform'))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='sgd')

# demonstration of verbose modes 1 and 2
model.fit(X_train, y_train, nb_epoch=3, batch_size=16, validation_split=0.1, verbose=1)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
37800/37800 [==============================] - 7s - loss: 0.0385 - acc.: 0.7258
Epoch 1
37800/37800 [==============================] - 8s - loss: 0.0140 - acc.: 0.9265
Epoch 2
10960/37800 [=======>......................] - ETA: 4s - loss: 0.0109 - acc.: 0.9420
'''

model.fit(X_train, y_train, nb_epoch=3, batch_size=16, verbose=2)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
loss: 0.0190 - acc.: 0.8750
Epoch 1
loss: 0.0146 - acc.: 0.8750
Epoch 2
loss: 0.0049 - acc.: 1.0000
'''

# demonstration of show_accuracy
model.fit(X_train, y_train, nb_epoch=3, batch_size=16, validation_split=0.1, show_accuracy=True, verbose=1)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
37800/37800 [==============================] - 7s - loss: 0.0385 - acc.: 0.7258 - val. loss: 0.0160 - val. acc.: 0.9136
Epoch 1
37800/37800 [==============================] - 8s - loss: 0.0140 - acc.: 0.9265 - val. loss: 0.0109 - val. acc.: 0.9383
Epoch 2
10960/37800 [=======>......................] - ETA: 4s - loss: 0.0109 - acc.: 0.9420
'''
```