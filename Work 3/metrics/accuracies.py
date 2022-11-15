def accuracy(labels, predictions):
    sum = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            sum += 1
    return (sum / len(labels)) * 100
