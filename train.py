import argparse
from time import time
import torch
from torchvision import datasets, transforms
import utility
import model_helper
import os

def gpu_check():
	print("PyTorch version is {}".format(torch.__version__))
	gpu_check = torch.cuda.is_available()

	if gpu_check:
		print("GPU Device available.")
	else:
		warnings.warn('No GPU found. Please use a GPU to train your neural network.')
		
	return gpu_check
	
def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help='Directory to training images')

    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    parser.add_argument('--arch', dest='arch', default='densenet161', action='store',choices=['vgg13', 'densenet161'], help='Architecture')

    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')

    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')

    parser.add_argument('--epochs', type=int, default=20, help='Epoch count')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')
    parser.set_defaults(gpu=False)

    return parser.parse_args()

def main():
	Input_aruguments = argument_parser()
	print("Chosen Learning rate is {}, Hidden Units is {} and Epochs are {}".format(Input_aruguments.learning_rate, Input_aruguments.hidden_units, Input_aruguments.epochs))

	batch_size = 64
	
	gpu_check = torch.cuda.is_available() and Input_aruguments.gpu
	if gpu_check:
		print("GPU Device available.")
	else:
		warnings.warn("No GPU found. Please use a GPU to train your neural network.")
		
	print("Data loading started.")
	train_dir = Input_aruguments.data_dir + '/train'
	valid_dir = Input_aruguments.data_dir + '/valid'
	test_dir = Input_aruguments.data_dir + '/test'

	data_transforms = {
    'training_sets' : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                                                            
    'validation_sets' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'testing_sets' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
	}

	# Load the datasets with ImageFolder
	image_datasets = {
		'training_sets':datasets.ImageFolder(train_dir, transform=data_transforms['training_sets']),
		'validation_sets':datasets.ImageFolder(valid_dir, transform=data_transforms['validation_sets']),
		'testing_sets':datasets.ImageFolder(test_dir, transform=data_transforms['testing_sets'])
	}

	# Using the image datasets and the transforms, define the dataloaders
	dataloaders = {
		'training_sets':torch.utils.data.DataLoader(image_datasets['training_sets'], batch_size, shuffle=True),
		'validation_sets':torch.utils.data.DataLoader(image_datasets['validation_sets'], batch_size, shuffle=True),
		'testing_sets':torch.utils.data.DataLoader(image_datasets['testing_sets'], batch_size, shuffle=True)
	}
	print("Data loading completed. Model creation in-progress, please wait.")
	model, optimizer, criterion = model_helper.create_model(Input_aruguments.arch,Input_aruguments.hidden_units,Input_aruguments.learning_rate,image_datasets['training_sets'].class_to_idx)
	
	print("Model creation completed. Moving to GPU if available, please wait.")
	if gpu_check:
		model.cuda()
		criterion.cuda()
	
	print("Training started, please wait it might take upto 5 mins.")
	model_helper.train(model, criterion, optimizer, Input_aruguments.epochs, dataloaders['training_sets'], dataloaders['validation_sets'], gpu_check)
	print("Training completed. Saving checkpoints, please wait.")
	model_helper.save_checkpoint(model, optimizer, batch_size, Input_aruguments.learning_rate ,Input_aruguments.arch, Input_aruguments.hidden_units, Input_aruguments.epochs)
	print("Saving checkpoints complete. Validating model, please wait.")
	test_loss, accuracy = model_helper.validate(model, criterion, dataloaders['testing_sets'],gpu_check)
	print("Validation Accuracy: {:.3f}".format(accuracy))
	image_path = 'flower_data/test/66/image_05582.jpg'
	print("Predication for: {}".format(image_path))
	probs, classes = model_helper.predict(image_path, model, gpu_check)
	print(probs)
	print(classes)
	 
	 
	 
	 
if __name__ == "__main__":
    main()