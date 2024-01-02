import pandas as pd
import numpy as np
import torch
from utils import Xy_ops,Sets_ops,Trainer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from gen import get_df

class Preprocessing:
    def normalization(df):
        for col in df.select_dtypes(include=np.number):
            if col !="ppv" and col!="frequency":
                np_df=df[col].to_numpy()
                df[col]=abs(np_df-np_df.mean())/np_df.std()
        return df 
    
    def preprocess(config):
        df=get_df()
        df=Preprocessing.normalization(df)
        print(df.head())
        trainset,valset,testset=Sets_ops.umbalanced_split(df)
        X_train,y_train=Xy_ops.xy_split(trainset,labels_names=["ppv","frequency"])
        X_val,y_val=Xy_ops.xy_split(valset,labels_names=["ppv","frequency"])
        X_test,y_test=Xy_ops.xy_split(testset,labels_names=["ppv","frequency"])
        X_train,y_train,trainloader=Xy_ops.xy_to_loader(X_train,y_train)
        X_val,y_val,valloader=Xy_ops.xy_to_loader(X_val,y_val)
        X_test,y_test,testloader=Xy_ops.xy_to_loader(X_test,y_test)
        return(X_train,y_train,X_val,y_val,X_test,y_test),(trainloader,valloader,testloader) 

class iron_grade_net(nn.Module):
    def __init__(self,layer_size1,layer_size2):
        super().__init__()
        self.fc1=nn.Linear(5,layer_size1)
        self.fc2=nn.Linear(layer_size1,layer_size2)
        self.fc3=nn.Linear(layer_size2,2)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

if __name__=="__main__":
    config={"data":r"./data/ore_grade_data.csv",
            "model_name":"ore_grade_net",
            "patience":10,
            "epoch":100,
            "min-delta":20,
            "verbose":1,
            "squared":False,
            "test_size":0.1,
            "val_size":0.3,
            "random_state":0,
            "shuffle":True}
    pd.set_option("display.max_columns",100)
    tensors,loaders=Preprocessing.preprocess(config)
    for tensor in tensors:
        print(tensor.size())
    device=torch.device("cuda")
    model=iron_grade_net(512,2048)
    model=Trainer.train(model=model,
                trainloader=loaders[0],
                valloader=loaders[1],
                criterion=nn.L1Loss(),
                optimizer=optim.Adam(model.parameters()),
                model_name=config["model_name"],
                device=device,
                 patience=config["patience"],
                epoch=config["epoch"],
                 min_delta=config["min-delta"],
                 verbose=config["verbose"])
    model.to('cpu')
    model=model.eval()
    with torch.no_grad():
        y_train_pred=model(tensors[0])
        y_val_pred=model(tensors[2])
        y_test_pred=model(tensors[4])
        ppv_train_mse=mean_squared_error(y_train_pred[0],tensors[1][0],squared=config["squared"])
        ppv_val_mse=mean_squared_error(y_val_pred[0],tensors[3][0],squared=config["squared"])
        ppv_test_mse=mean_squared_error(y_test_pred[0],tensors[5][0],squared=config["squared"])

        f_train_mse=mean_squared_error(y_train_pred[1],tensors[1][1],squared=config["squared"])
        f_val_mse=mean_squared_error(y_val_pred[1],tensors[3][1],squared=config["squared"])
        f_test_mse=mean_squared_error(y_test_pred[1],tensors[5][1],squared=config["squared"])

        print("ppv")
        print('------')
        print(ppv_train_mse)
        print(ppv_val_mse)
        print(ppv_test_mse)
        print("frequency")
        print('------------')
        print(f_train_mse)
        print(f_val_mse)
        print(f_test_mse)
   

