
import argparse
import glob
import os
import torch
from torchvision import transforms
from PIL import Image
from network import MLP
from torch import nn
# python create_db_mnist.py -d data/mini_mnist/train/ -g 0
def createDatabase(paths, args):
	# Create model
	model = MLP(args.unit, 28*28, 10)
	model.load_state_dict(torch.load(args.model))
	model = nn.Sequential(model.fc1, model.fc2)
	model.eval()

	# Set transformation
	#normalize = transforms.Normalize(mean=[0.], std=[0.5]) 学習する時normalizeはなかったので割愛
	data_preprocess = transforms.Compose([
		transforms.ToTensor(),
		nn.Flatten()])
	# Set model to GPU/CPU
	device = 'cpu'
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else "mps"
	model = model.to(device)
	# Get features
	print("device: ", device)
	with torch.no_grad():
		features = torch.cat(
			[model(data_preprocess(Image.open(path, 'r')).to(device)).to('cpu')
				for path in paths],
			dim = 0
		)
	# Show created dataset size
	print('dataset size : {}'.format(len(features)))
	return features

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(create database)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--dataset', '-d', default='default_dataset_path',
						help='Directory for creating database')
	parser.add_argument('--unit', '-u', type=int, default=1000,
						help='Number of units')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	args = parser.parse_args()

	data_dir = args.dataset

	# Get a list of pictures
	paths = glob.glob(os.path.join(data_dir, './*/*.png'))

	assert len(paths) != 0 
	# Create the database
	features = createDatabase(paths, args)
	# Save the data of database
	torch.save(features, 'result/feature_mlp.pt')
	torch.save(paths, 'result/path_mlp.pt')

if __name__ == '__main__':
	main()
