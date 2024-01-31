from PIL import Image 
import sys,os
import argparse
import pandas as pd
import numpy as np


user_data=pd.read_csv('users_data.csv')


if __name__=='__main__':
    # initialize the parser
    parser=argparse.ArgumentParser(
        description="Enter the User_id, Name and image path"
    )

    # Add the parameters
    parser.add_argument('User_id',help="Enter the User ID here",type=float)
    parser.add_argument('Name',help="Enter the Name here")
    parser.add_argument('Image',help="Enter the Image path")
    # parse the argument
    args=parser.parse_args()

    if (user_data['User_id']==args.User_id).any():
        print('Username already exits')
    else:
        print("User_id :",args.User_id)
        print("Name :",args.Name)
        print("Image path:",args.Image)
        new_row=[pd.Timestamp.now(),args.User_id,args.Name,args.Image]
        user_data.loc[len(user_data)] = new_row 
        print(user_data)
        user_data.to_csv('users_data.csv', index=False)

    
    image_path='./user_face_info/'   
    im1 = Image.open(user_data.iloc[len(user_data)-1]['Image Location'])
    im1.save(image_path+str(user_data.iloc[len(user_data)-1]['Username'])+".png") 

