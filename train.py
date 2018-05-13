#coding:utf-8
from keras.models import  Model,load_model
from keras.layers import  GlobalAveragePooling2D, Dense,Dropout,Concatenate,GlobalMaxPooling2D,Input
from keras.callbacks import ModelCheckpoint, History
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import SGD,Adam
# from keras.utils.vis_utils import plot_model
# import cv2
import numpy as np
from preprocess import FILENAME_V,FILENAME,generate_batch,generate_val_batch
from keras import backend as K

class TrainModel(object):
    def __init__(self):
        self.model_path = 'model.h5'
        self.continue_model_path = 'continue_model.h5'
        
    # def draw_model(self,model):
    #     plot_model(model, to_file='model.jpg', show_shapes=True)


    
    def inceptionResNetV2(self):
        base_model =InceptionResNetV2(weights=None, include_top=False)
        

        print("loading inception_resnet_v2")
        base_model.load_weights("inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")  
        print("done")

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x) #new FC layer, random init
        x = Dropout(0.5)(x)
        predictions = Dense(20, activation='softmax')(x) #new softmax layer
        
        model = Model(input=base_model.input, output=predictions)
        return model

    def inception(self):
        base_model = InceptionV3(weights="imagenet", include_top=False)

        # print("loading inceptionv3")
        # base_model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")  
        # print("done")

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x) #new FC layer, random init
        x = Dropout(0.5)(x)
        predictions = Dense(20, activation='softmax')(x) #new softmax layer
        model = Model(input=base_model.input, output=predictions)
        return model

   

    def train(self,model, batch_size=32, samples_per_epoch=200, epochs=5):  #32,1000,1
        
        optimizer=Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model_checkpoint = ModelCheckpoint(
                self.model_path, monitor='val_loss',verbose=1, save_best_only=True)

        model.fit_generator(
            generate_batch(FILENAME, batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=generate_val_batch(FILENAME_V,32),
            validation_steps=100,
            shuffle=True,
            callbacks=[model_checkpoint]
        )

        print('save to '+ self.model_path)



    def continue_train_sgd(self,filename,batch_size=32, samples_per_epoch=200, epochs=50):
        model=load_model(filename)

        
        epochs = 50
        learning_rate = 1e-4
        decay_rate = learning_rate / epochs
        momentum = 0.8

        optimizer=SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)  #继续训练选用sgd
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model_checkpoint = ModelCheckpoint(
                self.continue_model_path, monitor='val_loss',verbose=1, save_best_only=True)

        model.fit_generator(
           	generate_batch(FILENAME, batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=generate_val_batch(FILENAME_V,32),
            validation_steps=10,
            shuffle=True,
            callbacks=[model_checkpoint]
        )

        

        print('save to '+ self.continue_model_path)

    def eval(self,filename):  
        model =load_model(filename)
        g = generate_test_batch(FILENAME_V,10)
        result = model.evaluate_generator(g, steps=100)
        print("loss为"+result)
        return result

    def predict_top3_in_validation(self,filename):  #top3
        model =load_model(filename)
        s=0
        r=0
        for x,y in generate_val_batch(FILENAME_V,10):
            r=model.predict(x)
            rea=np.argmax(y,axis=0)
            for i in range(np.shape(r)[0]):
                pre = np.argsort(r[i])[-3:]
                if(pre[0]==rea[i] or pre[1]==rea[i] or pre[2]==rea[i]):
                    r+=1
                s+=1
        print("top3正确率"+str(r*100/s)+"%")
            
    def predict_top1_in_validation(self,filename):  #top3
        model =load_model(filename)
        s=0
        r=0
        for x,y in generate_val_batch(FILENAME_V,10):
            r=model.predict(x)
            rea=np.argmax(y,axis=0)
            for i in range(np.shape(r)[0]):
                pre = np.argsort(r[i])[-1:]
                if(pre[0]==rea[i]):
                    r+=1
                s+=1
   
        print("top1正确率"+str(r*100/s)+"%")


    # def predict_one(self,filename):
    #     model =load_model(filename)
    #     x, y = generate_batch(FILENAME_V).__next__()
    #     r =  model.predict(x)
    #     x *= 255
    #     x = x.astype('uint8')
    #     x = np.squeeze(x)
    #     print(y,r)
    #     im = cv2.imshow('x', x)
    #     cv2.waitKey(0)
    def test_one(self,filename):
        model =load_model(filename)
        x, y = generate_val_batch(FILENAME_V,10).next()
        r =  model.predict(x)
        # x *= 255
        # x = x.astype('uint8')
        # x = np.squeeze(x)
        print(y,r)

if __name__ == '__main__':
    T=TrainModel()
    # T.test_one(T.model_path)
    T.train(T.inception())
    # T.continue_train_sgd(T.model_path)