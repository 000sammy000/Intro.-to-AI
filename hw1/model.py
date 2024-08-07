import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):

        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four 
        attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)

        Xtrain=[]
        Ytrain=[]
        test=[]
        for i in range(len(train_data)):
            
            Xtrain.append(np.ravel(train_data[i][0]))
            Ytrain.append(train_data[i][1])


        self.x_train=np.asarray(Xtrain)
        self.y_train=np.asarray(Ytrain)
       
      
        

        Xtest=[]
        Ytest=[]
        for i in range(len(test_data)):
                Xtest.append(np.ravel(train_data[i][0]))
                Ytest.append(train_data[i][1])


        self.x_test=Xtest
        self.y_test=Ytest
            
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2-1)
        
        self.model = self.build_model(model_name)
        
    
    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        if model_name=="KNN":
            clf=KNeighborsClassifier(n_neighbors=3)
            return clf

        elif model_name=="RF":
            clf=RandomForestClassifier(n_estimators=90,max_depth=2,random_state=0)
            return clf

        else:
            clf=AdaBoostClassifier(n_estimators=3,random_state=0)
            return clf

        
    
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        
        self.model.fit(self.x_train, self.y_train)
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2-3)
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(self.y_test,y_pred))
    
    def classify(self, input):
        return self.model.predict(input)[0]
        

