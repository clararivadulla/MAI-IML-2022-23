import pandas as pd

def accuracy(labels, predictions):
    df = {'true labels':labels, 'predictions':predictions}
    df = pd.DataFrame(df)
    correct = sum(df['true labels']==df['predictions'])
    incorrect = len(labels) - correct
    share = (correct / len(labels))
    return correct, incorrect, share
