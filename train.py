from keras.models import  Model,load_model
from keras.layers import  GlobalAveragePooling2D, Dense,Dropout,Concatenate,GlobalMaxPooling2D,Input
from keras.callbacks import ModelCheckpoint, History
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.optimizers import SGD,Adam
from keras.utils.vis_utils import plot_model
import cv2
import numpy as np
from preprocess import FILENAME_V,FILENAME,generate_batch
from keras import backend as K

class TrainModel(object):
    def __init__(self):
        self.model_path = 'model.h5'
        self.continue_model_path = 'continue_model.h5'
        
    def draw_model(self,model):
        plot_model(model, to_file='model.jpg', show_shapes=True)

    def Multimodel(self,cnn_weights_path=False,class_num=20):  #cnn_weights_path 如果下载了weighths为“imagenet”的DenseNet201，Xception，就不用导入
        ''''' 
        获取densent201,xinception并联的网络 
        此处的cnn_weights_path是个列表是densenet和xception的卷积部分的权值 
        '''
        #需要挂在vpn
        input_layer=Input(shape=(224,224,3))  
        dense=DenseNet201(include_top=False,weights=None,input_shape=(224,224,3))  
        xception=Xception(include_top=False,weights=None,input_shape=(224,224,3))  
        #res=ResNet50(include_top=False,weights=None,input_shape=(224,224,3))  
      
        if cnn_weights_path==True:
            print("loading densenet")
            dense.load_weights("densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5")  
            print("done")
            print("loading xception")
            xception.load_weights("xception_weights_tf_dim_ordering_tf_kernels_notop.h5")
            print("done")

        

        for i,layer in  enumerate(dense.layers):  
            dense.layers[i].trainable=False  
        for i,layer in enumerate(xception.layers):  
            xception.layers[i].trainable=False  
        #for i,layer in enumerate(res.layers):  
        #   res.layers[i].trainable=False  
      
   
        dense=dense(input_layer)  
        xception=xception(input_layer)  
      
      
        #对dense_121和xception进行全局最大池化  
        top1_model=GlobalMaxPooling2D(data_format='channels_last')(dense)  
        top2_model=GlobalMaxPooling2D(data_format='channels_last')(xception)  
        #top3_model=GlobalMaxPool2D(input_shape=res.output_shape)(res.outputs[0])  
          
        # print(top1_model.shape,top2_model.shape)  
        #把top1_model和top2_model连接起来  
        t=Concatenate(axis=1)([top1_model,top2_model])  
        #第一个全连接层  
        top_model=Dense(units=512,activation="relu")(t)  
        top_model=Dropout(rate=0.5)(top_model)  
        top_model=Dense(units=class_num,activation="softmax")(top_model)  
          
        model=Model(inputs=input_layer,outputs=top_model)  
        

        return model  

    def train(self,model, batch_size=8, samples_per_epoch=500, epochs=16):  #8,2000,5
        # history = History()
        optimizer=Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model_checkpoint = ModelCheckpoint(
                self.model_path, monitor='val_loss',verbose=1, save_best_only=True)

        model.fit_generator(
            generate_batch(FILENAME, batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=generate_arrays_from_file(FILENAME_V),
            validation_steps=100,
            shuffle=True,
            callbacks=[model_checkpoint]
        )
        # model.save('vgg16.h5' + '.' + str(history.history['val_loss'][-1]))
        print('save to '+ self.model_path)

    def continue_train(self,filename,batch_size=8, samples_per_epoch=500, epochs=16):
        model=load_model(filename)

        optimizer=SGD()  #继续训练选用sgd
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model_checkpoint = ModelCheckpoint(
                self.continue_model_path, monitor='val_loss',verbose=1, save_best_only=True)

        model.fit_generator(
           	generate_batch(FILENAME, batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=generate_arrays_from_file(FILENAME_V),
            validation_steps=100,
            shuffle=True,
            callbacks=[model_checkpoint]
        )
        print('save to '+ self.continue_model_path)

    def eval(self,filename):  
        model =load_model(filename)
        g = generate_batch(FILENAME_V,10)
        result = model.evaluate_generator(g, steps=100)
        print("loss为"+result)
        return result

    def predict_top3_in_validation(self,filename):  #top3
        model =load_model(filename)
        s=0
        r=0
        for x,y in generate_batch(FILENAME_V,10):
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
        for x,y in generate_batch(FILENAME_V,10):
            r=model.predict(x)
            rea=np.argmax(y,axis=0)
            for i in range(np.shape(r)[0]):
            	pre = np.argsort(r[i])[-1:]
            	if(pre[0]==rea[i]):
                    r+=1
                s+=1
   
        print("top1正确率"+str(r*100/s)+"%")


    def predict_one(self,filename):
        model =load_model(filename)
        x, y = generate_batch(FILENAME_V).__next__()
        r =  model.predict(x)
        x *= 255
        x = x.astype('uint8')
        x = np.squeeze(x)
        print(y,r)
        im = cv2.imshow('x', x)
        cv2.waitKey(0)

if __name__ == '__main__':
    T=TrainModel()
    T.draw_model(T.Multimodel(cnn_weights_path=True))