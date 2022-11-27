def accuracy(labels, predictions):
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct += 1
    incorrect = len(labels) - correct
    percentage = (correct / len(labels)) * 100
    return correct, incorrect, percentage
