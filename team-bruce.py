"""						
This script is the starter script that the user needs to run, which will set up 
the dataset, call the corresponding models that the user chooses, and write the
results into the 'results' directory.
---------------------------
Author : Aashish Yadavally
"""


import site
import os
site.addsitedir(os.path.dirname(os.path.realpath(__file__))) 


import argparse
from src.data_loader import DataLoader
from src.models.unet import UNet
from src.models.unet_postprocessing import histogram_binning
from src.models.optical_flow import OpticalFlow
from src.models.variance import Variance
from src.models.lstm_unet import LstmUnet
import logging


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Team Bruce: Cilia Segmentation')
parser.add_argument('--model', type=str, choices=['unet', 'variance', 'optical-flow', 'lstm-unet'], 
	default='unet', help = 'model to use for cilia segmentation')
parser.add_argument('--url', type=str, default='gs://uga-dsp/project2', 
	help='google storage bucket url')
parser.add_argument('--data', type=str, default='cilia_dataset', 
	help='name of folder to download dataset into')
args = parser.parse_args()
model = args.model

dl = DataLoader(args.url, args.data)
if model == 'unet':
	UNet(model)
	logger.info('Successfully trained U-Net model on Cilia Dataset!')
	histogram_binning(model)
	logger.info('Prediction masks have been saved in \'results/unet/predictions\' directory.')
elif model == 'lstm_unet':
	LstmUnet(model)
	logger.info('Prediction masks have been finished.')
elif model == 'variance':
	Variance(model)
	print('Prediction masks have been saved in \'results/variance/predictions\' directory.')
elif model == 'optical-flow':
	OpticalFlow(model)
	logger.info('Prediction masks have been saved in \'results/optical-flow/predictions\' directory.')
else:
	logger.error('Invalid choice entered for model!')
