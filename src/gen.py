import pandas as pd
import random 
import numpy as np
from utils import Gen_synthetic_data

def get_df():
    blastholes=[5,5,5,5,5,5,5,5,15,15,15,15,40,40,40]
    diameter=[105,105,105,105,140,140,140,140,105,105,140,140,105,140,105]
    spacing=[4.0,4.5,5.0,5.5,4.0,4.5,5.0,5.5,4.0,5.0,4.0,5.0,4.5,4.5,4.5]
    plastic_explo=[25,25,25,25,25,25,25,25,25,25,25,25,35,25,25]
    anfo_explo=[200,200,200,200,200,200,200,200,200,200,200,200,150,200,180]
    ppv=[1.5,1.5,6.1,2.0,1.9,1.8,5.5,6.8,1.3,1.9,1.5,1.3,1.3,1.7,2.0]
    frequency=[18,14,2,12,10,20,10,11,19,17,17,17,16,16,13]
    frames=zip(blastholes,diameter,spacing,plastic_explo,anfo_explo,ppv,frequency)
    df=pd.DataFrame(frames,columns=["blastholes","diameter",
                                    "spacing","plastic_explosive",
                                    "anfo_explosive","ppv","frequency"])
    df=Gen_synthetic_data.multi_target_gen_balanced_data(df,target_names=["ppv","frequency"],
                                                         gen_nbr_rows=[20,20])
    return df

     
if __name__=='__main__':
    df=get_df()
    print(df)
    print(df.isna().sum())
    print(df["ppv"].value_counts())
    print(df["frequency"].value_counts())

 
