import numpy as np
import keras.models
from keras.models import model_from_json
#from scipy.imageio import imread, imresize, imshow
import tensorflow as tf

def init():
    json_file = open('new_model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("new_model.h5")   # load weights into new model
    print("Loaded Model from disk")
    #Compile and Evaluate model
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #score = loaded_model.evaluate_generator(validation_generator, nb_validation_samples, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    graph = tf.get_default_graph()

    return loaded_model,graph
