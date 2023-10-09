
import argparse
import torch
from torchvision import transforms
from PIL import Image
from network_db_overwritten import Vgg16
from torch import nn
from network import MLP
# python search_mnist.py -i data/mini_mnist/test/0/3.png
def search(args):
	# Create model
	model = MLP(args.unit, 28*28, 10)
	model.load_state_dict(torch.load(args.model))
	model = nn.Sequential(model.fc1, model.fc2)
	model.eval()
	# Set transformation
	#normalize = transforms.Normalize(mean=[0.], std=[0.5])
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
	with torch.no_grad():
		src_feature = model(data_preprocess(Image.open(args.input, 'r')).to(device)).to('cpu')
	# Load database
	print("src_feature's shape: ", src_feature.shape)
	paths = torch.load(args.paths)
	features = torch.load(args.features)
	assert args.k <= len(paths)
	assert len(features) == len(paths)
	# Calculate distances
	distances = torch.tensor(
		[torch.norm(src_feature - feature)
			for feature in features]
	) #差のnormを計算している。
	_, indices = torch.topk(distances, args.k, largest=False)
	# Show results
	for i in indices:
		print(paths[i])

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(search image)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--input', '-i', default='default_source_image_path',
						help='path to database features')
	parser.add_argument('--features', '-f', default='result/feature_mlp.pt',
						help='path to database features')
	parser.add_argument('--paths', '-p', default='result/path_mlp.pt',
						help='path to database paths')
	parser.add_argument('--k', '-k', type=int, default=5,
						help='find num')
	parser.add_argument('--unit', '-u', type=int, default=1000,
						help='Number of units')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	args = parser.parse_args()

	search(args)

if __name__ == '__main__':
	main()