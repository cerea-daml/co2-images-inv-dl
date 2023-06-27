from tensorflow import keras
import tensorflow as tf
import numpy as np

# polynome
def P(x):
    a = 0.5
    b = 1.5
    c = 2
    d = 0.3
    return a*x[:,0]*x[:,1]+b*x[:,0]+c*x[:,1]+d

# samples
N = 12
X_0 = np.random.randn(N,1)
X_1 = np.random.randn(N,1)
Y = P(np.concatenate((X_0, X_1), axis=1))

# keras polynomial model
def polynomial_model():
    X_0 = keras.layers.Input(shape=(1))
    X_1 = keras.layers.Input(shape=(1))
    a = keras.layers.Dense(1, use_bias=False, name="a")(keras.layers.Multiply()([X_0, X_1]))
    b = keras.layers.Dense(1, use_bias=False, name="b")(X_0)
    c = keras.layers.Dense(1, use_bias=True, name="c_d")(X_1)
    
    output = keras.layers.Add()([a,b,c])
    
    model = keras.Model([X_0, X_1],output)
    
    return model

model = polynomial_model()
model.summary()
model.compile(loss=keras.losses.mean_squared_error)
model.fit([X_0, X_1], Y, epochs=3)

# 
for layer in model.layers: 
    print(layer.get_weights())
    
# compare output du polynôme originel avec output du modèle
Z = np.array([0.3, 0.4])
Z_0 = np.expand_dims(np.array([Z[0]]), axis=0)
Z_1 = np.expand_dims(np.array([Z[1]]), axis=0)

print("poly originel:", P(np.concatenate((Z_0,Z_1), axis=1)))
print("model:", model([Z_0, Z_1]))