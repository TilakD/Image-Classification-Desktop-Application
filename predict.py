import argparse
from time import time
import json
import torch
import utility
import model_helper

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str,help='Input image')

    parser.add_argument('checkpoint', type=str, help='Model checkpoint file to use for prediction')

    parser.add_argument('--top_k', type=int, default=3, help='Return top k most likely classes')

    parser.add_argument('--category_names', type=str, help='Mapping file used to map categories to real names')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for prediction')
    parser.set_defaults(gpu=False)

    return parser.parse_args()	


def get_title(label, cat_to_name):
    try:
        return cat_to_name[label]
    except KeyError:
        return "unknown label"


def main():
	Input_aruguments = argument_parser()
	print("Loading checkpoints in-progress.")
	model = model_helper.load_checkpoint(Input_aruguments.checkpoint)
	print("Loading checkpoints completed. Checking for GPU, please wait.")
	gpu_check = torch.cuda.is_available() and Input_aruguments.gpu
	if gpu_check:
		model.cuda()
		print("GPU Device available.")
	else:
		warnings.warn('No GPU found. Please use a GPU to train your neural network.')
	use_mapping_file = False
	if Input_aruguments.category_names:
		with open(Input_aruguments.category_names, 'r') as f:
			cat_to_name = json.load(f)
			use_mapping_file = True
	print("Prediction in-progress. Please wait.")
	probs, classes = model_helper.predict(Input_aruguments.input, model, gpu_check, Input_aruguments.top_k)


	print("\nTop {} Classes predicted for '{}':".format(len(classes), Input_aruguments.input))
	if use_mapping_file:
		print("\n{:<30} {}".format("Flower", "Probability"))
		print("{:<30} {}".format("------", "-----------"))
	else:
		print("\n{:<10} {}".format("Class", "Probability"))
		print("{:<10} {}".format("------", "-----------"))

	for i in range(0, len(classes)):
		if use_mapping_file:
			print("{:<30} {:.2f}".format(
				get_title(classes[i], cat_to_name), probs[i]))
		else:
			print("{:<10} {:.2f}".format(classes[i], probs[i]))

	
if __name__ == "__main__":
    main()
