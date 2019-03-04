import argparse
from src.data_loader import DataLoader
from src.models.unet import UNet
from src.models.unet_postprocessing import histogram_binning
from src.models.optical_flow import OpticalFlow
import logging


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Team Bruce: Cilia Segmentation')
parser.add_argument('--model', type=str, choices=['unet', 'variance', 'optical-flow', 'robust-pca'], 
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
	print('Prediction masks have been saved in \'results/unet/predictions\' directory.')
elif model == 'lstm_unet':
    LSTM_UNET(model)
    print('Prediction masks have been finished.')
elif model == 'variance':
	VARIANCE(model)
    print('Prediction masks have been saved in \'results/variance/predictions\' directory.')
elif model == 'optical-flow':
	OpticalFlow(model)
	logger.info('Prediction masks have been saved in \'results/optical-flow/predictions\' directory.')
elif model == 'robust-pca':
	pass
