import os
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from natsort import natsorted
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
import random 

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

class Xy_ops:
    def xy_to_loader(X,y,batch_size=4,shuffle=False,return_tensor=True):
        X=torch.from_numpy(X.to_numpy()).float()
        if len(y.shape)<2:
            y=np.reshape(y.to_numpy(),(y.shape[0],1))
        else:
            y=y.to_numpy()
        y=torch.from_numpy(y).float()
        Xy_set=TensorDataset(X,y)
        Xy_loader=DataLoader(Xy_set,batch_size,shuffle)
        if return_tensor==False:
            return Xy_loader 
        elif return_tensor==True:
            return X,y,Xy_loader

    def xy_split(dataframe,labels_names:list):
        X=dataframe.drop(labels_names,axis=1)
        y=dataframe[labels_names]
        return X,y

class Sets_ops:
    def umbalanced_split(df,split_ratio=(0.1,0.3),shuffle=False):
        trainset,testset=train_test_split(df,test_size=split_ratio[0],random_state=0,shuffle=shuffle)
        trainset,valset=train_test_split(trainset,test_size=split_ratio[1],random_state=0,shuffle=shuffle)
        return trainset,valset,testset

class Gen_synthetic_data:
    def gen(df,gen_sample=True,sample_nbr=10):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        model=GaussianCopulaSynthesizer(metadata)
        model.fit(df)
        if gen_sample==True:
            gen_sample=model.sample(num_rows=sample_nbr)
            new_df=pd.concat([df,gen_sample],axis=0)
            return model,new_df
        else:
            return model

    def multi_target_gen_balanced_data(df,
                          target_names:list,
                        gen_nbr_rows:list,
                          frac=1,
                          random_state=0):
        datasets=[]
        target_names=target_names
        nbr=0
        for target_name in target_names:
            target_to_remove=[]
            for item in target_names:
                if item!=target_name:
                    target_to_remove.append(target_names.pop(target_names.index(item)))
            for item in target_to_remove:
                target_names.append(item)
            the_df=df.drop(target_to_remove,axis=1)
            model=Gen_synthetic_data.gen(the_df,gen_sample=False)
            condition=[]
            unique_target=dict(the_df[target_name].value_counts())
            for key in unique_target:
                value=unique_target[key]
                nbr_rows=gen_nbr_rows[nbr]-value
                if nbr_rows >0:
                    condition.append(Condition(num_rows=nbr_rows,column_values={target_name:key}))
            gen_df=model.sample_from_conditions(condition)
            gen_df=pd.concat([the_df,gen_df],axis=0)
            datasets.append(gen_df)
            nbr=nbr+1
        new_df=datasets[0].merge(right=datasets[1])
        new_df=new_df.sample(frac=frac,random_state=random_state)
        return new_df

class Trainer:
    def create_dir(model_name):
        base_dir=os.getcwd()
        model_dir="{}/model".format(base_dir)
        model_path="{}/{}.pt".format(model_dir,model_name)
        if os.path.exists(model_dir)==False:
            os.mkdir(model_dir)
        return model_path

    def trainset_trainer(model,trainloader,optimizer,criterion,device):
        train_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            # forward+backward+optimize
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        return train_loss

    def valset_trainer(model,valloader,optimizer,criterion,device):
        val_loss=0.0
        for i,data in enumerate(valloader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            # forward+backward+optimize
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            val_loss+=loss.item()
        return val_loss

    def train (model,
               trainloader,
               valloader,
               criterion,
               optimizer,
               model_name,
               device,
               epoch=10,
               patience=3,
               min_delta=10,
               verbose=1):
        counter=0
        model=model.to(device)
        min_val_loss=float('inf')
        model_path=Trainer.create_dir(model_name)
        if verbose==2:print("training...")
        for epoch in range(1,epoch):
            train_loss=Trainer.trainset_trainer(model,trainloader,optimizer,criterion,device)
            val_loss=Trainer.valset_trainer(model,valloader,optimizer,criterion,device)
            if verbose==1:
                print("[epoch {}]: train_loss={}||val_loss={}".format(epoch,train_loss,val_loss))
            if val_loss<min_val_loss:
                min_val_loss=val_loss
                torch.save(model.state_dict(),model_path)
                counter=0
            elif (val_loss>min_val_loss+min_delta):
                counter+=1
                if counter>=patience:
                    break
                else:
                    pass
        if verbose==2: print("[epoch {}]: train_loss={}||val_loss={}".format(epoch,train_loss,val_loss))
        return model
                    


