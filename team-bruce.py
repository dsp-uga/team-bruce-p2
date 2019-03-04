import argparse
from src.data_loader import DataLoader
from src.models.unet import UNet

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
	UNet()
elif model == 'variance':
	pass
elif model == 'optical-flow':
	pass
elif model == 'robust-pca':
	pass



