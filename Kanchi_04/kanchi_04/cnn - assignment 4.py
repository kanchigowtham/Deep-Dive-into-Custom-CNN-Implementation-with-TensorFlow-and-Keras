# Kanchi, Gowtham Kumar
# 1002-044-003
# 2022_11_13
# Assignment-04-01


# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

class EpochLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.epoch_losses.append(logs.get('loss'))

class CNN(object):
    def __init__(self):
        self.model = keras.Sequential()
        self.metrics = []


    def add_input_layer(self, shape=(2,),name="" ):
        
        self.input_dimensions = shape
        


    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
       

        if not len(self.model.layers)==0:
            self.model.add(keras.layers.Dense(num_nodes,activation=activation.lower(),name=name,trainable=trainable))

        else:
            if type(self.input_dimensions) != type(()):
                self.model.add(
                    keras.layers.Dense(num_nodes, activation=activation.lower(), input_shape=(self.input_dimensions,),
                                       name=name, trainable=trainable))
            else:
                self.model.add(
                    keras.layers.Dense(num_nodes, activation=activation.lower(), input_shape=self.input_dimensions,
                                       name=name, trainable=trainable))
              

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        
        if not len(self.model.layers)==0:
            self.model.add(keras.layers.Conv2D(num_of_filters,kernel_size, strides=(strides,strides), padding = padding, activation=activation.lower(),name=name))

        else:
            self.model.add(keras.layers.Conv2D(num_of_filters,kernel_size, strides=(strides,strides), padding = padding, activation=activation.lower(),input_shape=self.input_dimensions,name=name))
            


    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        
        if not len(self.model.layers)==0:
            self.model.add(keras.layers.MaxPooling2D(pool_size, strides=(strides,strides),padding=padding,name=name))

        else:
            self.model.add(keras.layers.MaxPooling2D(pool_size, strides=(strides,strides),padding=padding,input_shape=self.input_dimensions,name=name))

            


    def append_flatten_layer(self,name=""):
       
        self.model.add(keras.layers.Flatten())


    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
       
        if layer_numbers:
            if type(layer_numbers)!=type([]):
                layer_numbers = [layer_numbers]
            for layer in layer_numbers:
                if layer>=0:
                    self.model.layers[layer-1].trainable = trainable_flag
                else:
                    self.model.layers[layer].trainable = trainable_flag

            else:
                if type(layer_names) != type([]):
                    layer_names = [layer_names]
                for layer in layer_names:
                    for i in range(len(self.model.layers)):
                        if self.model.layers[i].name == layer:
                            self.model.layers[i].trainable = trainable_flag
    

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        
        if  layer_number:
            if layer_number>=0:
                weight_bias = self.model.layers[layer_number-1].get_weights()
            else:
                weight_bias = self.model.layers[layer_number].get_weights()
            if weight_bias ==[]:
                return None
            return weight_bias[0]
        else:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        weight_bias = self.model.layers[i].get_weights()
                        if weight_bias ==[]:
                            return None
                        return weight_bias[0]
                  


    def get_biases(self,layer_number=None,layer_name=""):
        
        if  layer_number:
            if layer_number>=0:
                weight_bias = self.model.layers[layer_number-1].get_weights()
            else:
                weight_bias = self.model.layers[layer_number].get_weights()
            if weight_bias ==[]:
                return None
            return weight_bias[1]

        else:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        weight_bias = self.model.layers[i].get_weights()
                        if weight_bias ==[]:
                            return None
                        return weight_bias[1]
               

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        
        if layer_number:
            if layer_number>=0:
                delta_weight_bias =  self.model.layers[layer_number-1].get_weights()[1]
                self.model.layers[layer_number-1].set_weights([weights,delta_weight_bias])
            else:
                delta_weight_bias =  self.model.layers[layer_number].get_weights()[1]
                self.model.layers[layer_number].set_weights([weights,delta_weight_bias])

        else:
            for i in range(len(self.model.layers)):
                if self.model.layers[i].name == layer_name:
                    delta_weight_bias = self.model.layers[i].get_weights()[1]
                    self.model.layers[i].set_weights([weights, delta_weight_bias])
          


    def set_biases(self,biases,layer_number=None,layer_name=""):
       
        if  layer_number:
            if layer_number>=0:
                delta_weight_bias =  self.model.layers[layer_number-1].get_weights()[0]
                self.model.layers[layer_number-1].set_weights([delta_weight_bias,biases])
            else:
                delta_weight_bias =  self.model.layers[layer_number].get_weights()[0]
                self.model.layers[layer_number].set_weights([delta_weight_bias,biases])

        else:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        delta_weight_bias =  self.model.layers[i].get_weights()[0]
                        self.model.layers[i].set_weights([delta_weight_bias,biases])
                   


    def remove_last_layer(self):
        
        n = self.model.layers
        self.model = keras.Sequential(layers=n[:-1])
        return n[-1]


    def load_a_model(self,model_name="",model_file_name=""):
       
        if model_name:
            if model_name.lower() == "vgg19":
                self.model = keras.Sequential(layers=keras.applications.vgg19.VGG19().layers)
            elif model_name.lower() == "vgg16":
                self.model = keras.Sequential(layers=keras.applications.vgg16.VGG16().layers)
        else:
            self.model = keras.models.load_model(model_file_name)

        return self.model


    def save_model(self,model_file_name=""):
       
        self.model.save(model_file_name)
   


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
       
        if loss == "hinge":
            self.loss = keras.losses.hinge()
        elif loss == "SparseCategoricalCrossentropy":
            self.loss = keras.losses.SparseCategoricalCrossentropy()
        elif loss == "MeanSquaredError":
            self.loss = keras.losses.mean_squared_error()
            

    def set_metric(self,metric):
       
        self.metrics.append(metric)
     

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        
        if optimizer == "Adagrad":
            self.optimizer=keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer == "SGD":
            self.optimizer=keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        elif optimizer =="RMSprop":
            self.optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate)
     

    def predict(self, X):
        
        return self.model.predict(X)


    def evaluate(self,X,y):
        
        loss, metrics = self.model.evaluate(X,y)
        return loss, metrics
 

    def train(self, X_train, y_train, batch_size, num_epochs):
       
        losses = EpochLoss()
        self.model.compile(optimizer=self.optimizer,loss = self.loss, metrics=self.metrics)
        self.model.fit(X_train, y_train, batch_size = batch_size, epochs = num_epochs,callbacks=[losses])
        return losses.epoch_losses

if __name__ == "__main__":

    my_cnn=CNN()
    print(my_cnn)
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    weights=my_cnn.get_weights_without_biases(layer_number=0)
    biases=my_cnn.get_biases(layer_number=0)
    print("w0",None if weights is None else weights.shape,type(weights))
    print("b0",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=1)
    biases=my_cnn.get_biases(layer_number=1)
    print("w1",None if weights is None else weights.shape,type(weights))
    print("b1",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=2)
    biases=my_cnn.get_biases(layer_number=2)
    print("w2",None if weights is None else weights.shape,type(weights))
    print("b2",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=3)
    biases=my_cnn.get_biases(layer_number=3)
    print("w3",None if weights is None else weights.shape,type(weights))
    print("b3",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=4)
    biases=my_cnn.get_biases(layer_number=4)
    print("w4",None if weights is None else weights.shape,type(weights))
    print("b4",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_number=5)
    biases = my_cnn.get_biases(layer_number=5)
    print("w5", None if weights is None else weights.shape, type(weights))
    print("b5", None if biases is None else biases.shape, type(biases))

    weights=my_cnn.get_weights_without_biases(layer_name="input")
    biases=my_cnn.get_biases(layer_number=0)
    print("input weights: ",None if weights is None else weights.shape,type(weights))
    print("input biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv1")
    biases=my_cnn.get_biases(layer_number=1)
    print("conv1 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="pool1")
    biases=my_cnn.get_biases(layer_number=2)
    print("pool1 weights: ",None if weights is None else weights.shape,type(weights))
    print("pool1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv2")
    biases=my_cnn.get_biases(layer_number=3)
    print("conv2 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv2 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="flat1")
    biases=my_cnn.get_biases(layer_number=4)
    print("flat1 weights: ",None if weights is None else weights.shape,type(weights))
    print("flat1 biases: ",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense1")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense1 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense1 biases: ", None if biases is None else biases.shape, type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense2")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense2 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense2 biases: ", None if biases is None else biases.shape, type(biases))


