from src.classifier.mnist_classifier import *

# ------------------------------- Run all experiments and write them to app/data/experiment-results.txt ------------------------------------------------------ #

if __name__ == '__main__':

    # Load dataset and split it to train set and test set
    xtrain, ytrain, xtest, ytest = load_dataset()

    # Evaluate different network configurations as requested in DATA ANALYSIS_ProjectΣεπτ2020.pdf. Result data is written into app/data/experiment-results.txt
    evaluate_nets(xtrain, ytrain, xtest, ytest, eval_layers=True, eval_neurons=True, eval_epochs=True, eval_training_size=True, n_nets=2)

