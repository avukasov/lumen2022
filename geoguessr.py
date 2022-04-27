import os
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image


def load_images(img_path_dir, filenames):
    images = [np.array(Image.open(r""+img_path_dir + "/" + img)) for img in filenames]

    d_transformer = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=((256, 256))),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
    
    images = torch.stack([d_transformer(image) for image in images])

    if len(filenames) == 1:
        images = torch.unsqueeze(images, 0) # add one dimension as batch
        
    return images

class Geoguesser_CNN(torch.nn.Module):
    def __init__(self, input_shape=(16, 3, 256, 256), num_classes=8):
        super(Geoguesser_CNN, self).__init__()

        self.input_shape = input_shape  # (batch_size, 3, 256, 256)
        self.num_classes = num_classes

        self.resnet = models.resnet50(pretrained=True)

        self.linear1 = nn.Linear(1000, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128,
                                 num_classes)
        
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)

        x = self.drop(self.relu(self.linear1(x)))  
        x = self.drop(self.relu(self.linear2(x)))
        x = self.softmax(self.linear3(x))

        return x

class FinalModel():
    def __init__(self, MODELDIR):
        self.device = torch.device('cpu')
        self.first_layer_model = Geoguesser_CNN(num_classes=8)
        self.model_0 = Geoguesser_CNN(num_classes=10)
        self.model_1 = Geoguesser_CNN(num_classes=11)
        self.model_2 = Geoguesser_CNN(num_classes=14)
        self.model_3 = Geoguesser_CNN(num_classes=7)
        self.model_4 = Geoguesser_CNN(num_classes=14)
        self.model_5 = Geoguesser_CNN(num_classes=15)
        self.model_6 = Geoguesser_CNN(num_classes=12)
        self.model_7 = Geoguesser_CNN(num_classes=15)
        self.load_model(MODELDIR)
        self.second_layer_models_dict = {
                    0: self.model_0,
                    1: self.model_1,
                    2: self.model_2,
                    3: self.model_3,
                    4: self.model_4,
                    5: self.model_5,
                    6: self.model_6,
                    7: self.model_7
                }
        self.prediction_df = pd.read_csv(MODELDIR + '/' +'predicted_coordinates_by_subclass.csv')

    def load_model(self, MODELDIR):
        self.first_layer_model.load_state_dict(torch.load(MODELDIR + '/' + 'first_layer_final_model.pth', map_location=self.device))
        self.model_0.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_0.pth', map_location=self.device)),
        self.model_1.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_1.pth', map_location=self.device)),
        self.model_2.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_2.pth', map_location=self.device)),
        self.model_3.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_3.pth', map_location=self.device)),
        self.model_4.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_4.pth', map_location=self.device)),
        self.model_5.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_5.pth', map_location=self.device)),
        self.model_6.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_6.pth', map_location=self.device)),
        self.model_7.load_state_dict(torch.load(MODELDIR + '/' +'second_layer_final_model_subclass_7.pth', map_location=self.device))

    # predict
    def predict(self, images):
        with torch.no_grad():
            images = images.squeeze(dim=0)
            
            self.first_layer_model.eval()
            first_layer_outputs = self.first_layer_model(images) # there are 4 images (considered as batch of 4) put through model 

            first_layer_general_output = torch.unsqueeze(torch.sum(first_layer_outputs, 0), dim=0)
            first_layer_pred = first_layer_general_output.data.max(1, keepdim=True)[1] #  gleda se najvjetojatnija klasa
            #print(first_layer_pred.item(), "output prvog sloja")

            second_layer_model = self.second_layer_models_dict[first_layer_pred.item()]
            second_layer_model.eval()

            second_layer_outputs = second_layer_model(images) # there are 4 images (considered as batch of 4) put through model 

            second_layer_general_output = torch.unsqueeze(torch.sum(second_layer_outputs, 0), dim=0)
            final_class_prediction = second_layer_general_output.data.max(1, keepdim=True)[1]  #  gleda se najvjetojatnija klasa 

            #print(final_class_prediction.item(), "output drugog sloja")

            ## procitaj koordinate iz csv-a

            # csv je negdje gore ucitan
            df_subclass = self.prediction_df.loc[(self.prediction_df['Subclass_label']== final_class_prediction.item()), ['Subclass_' + str(first_layer_pred.item())]] # daj mi one redove gdje mi je Subclass_label jednak i i ispisi mi samo stupac subclass_0
            latitude, longitude = np.array(df_subclass)[0][0][1:-2].split()

            return float(latitude), float(longitude)
