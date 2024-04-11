def map_to_age_band(age):
    return (age // 5) * 5

# def calculate_accuracy(predictions, labels):
#     correct = 0
#     total = len(labels)
#     for pred, label in zip(predictions, labels):
#         pred_band = map_to_age_band(pred)
#         label_band = map_to_age_band(label)
#         if pred_band == label_band:
#             correct += 1
#     accuracy = (correct / total) * 100
#     return round(accuracy, 2)

def calculate_accuracy(predictions, labels, tolerance):
    correct = 0
    total = len(labels)
    for pred, label in zip(predictions, labels):
        if abs(pred - label) <= tolerance:
            correct += 1
    accuracy = (correct / total) * 100
    return round(accuracy, 2)