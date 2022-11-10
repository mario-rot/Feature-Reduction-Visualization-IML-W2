import argparse

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-ds", "--DataSet", help = "['iris', 'cmc', 'pen-based', 'vowel']", default='iris', type=str)
parser.add_argument("-dr", "--DimReduct", help = "['pca','feat_agg']", default='pca', type=str)
parser.add_argument("-cl", "--ClustAlg", help = "['kmeans','agg_cl']", default='kmeans', type=str)
parser.add_argument("-vt", "--VisTech", help = "['pca', 't-sne']", default='pca', type=str)
 
args = parser.parse_args()

run = neptune.init(
    project="mario.rosas/CI-MLP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MmQyYTM0OS0zNzg0LTRiM2UtOWUyNi0wYTU1NzcxNmJiODkifQ==",
)  # your credentials

params = {"epochs":args.Epochs,
          "learning_rate": args.LearningRate, 
          "optimizer": args.Optimizer,
          "momentum": args.Momentum,
          "hidden_activations":args.HiddenAct,
          "output_activations":args.OutputAct,
          "hidden_units":args.HiddenUnits,
          "subsets_distribution":args.DataSplit,
          "loss_fuction": args.Loss, 
          "tolerance": args.Tolerance}