
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import jsonlines
import pandas as pd
import numpy as np

def json2df(path):
    s1=[]
    s2=[]
    label=[]
    with open(path, mode='r', encoding="utf8") as f:
        reader = jsonlines.Reader(f)
        for instance in reader:
            s1.append(instance['sentence1'])
            s2.append(instance['sentence2'])
            label.append(int(instance['gold_label']))
    return pd.DataFrame({'s1':s1, 's2':s2, 'label':label})

def Metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='macro')
    # target_names = ['class_0', 'class_1']
    report = classification_report(y_true, y_pred, digits=3)

    print('Accuracy: {:.1%}\nPrecision: {:.1%}\nRecall: {:.1%}\nF1: {:.1%}'.format(accuracy, macro_precision,
                                           macro_recall, weighted_f1))
    print("classification_report:\n")
    print(report)
    

def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):	
        enc_masks[e_id, :src_len] = 1  
    return enc_masks								

def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask
               
def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def train(model, dataloader, optimizer, criterion, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (q, _, h, _, label) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        q1 = q.to(device)
        q2 = h.to(device)
        labels = label.to(device)
        optimizer.zero_grad()
        logits, probs = model(q1, q2)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
    
def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    predictions = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (q, _, h, _, label) in dataloader:
            # Move input and output data to the GPU if one is used.
            q1 = q.to(device)
            q2 = h.to(device)
            labels = label.to(device)
            logits, probs = model(q1, q2)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
            predictions.extend(probs.cpu().numpy())
    predictions = np.array(predictions).argmax(axis = 1)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, predictions

def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    predictions = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (q, _, h, _, label) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            q1 = q.to(device)
            q2 = h.to(device)
            labels = label.to(device)
            _, probs = model(q1, q2)
            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start
            predictions.extend(probs.cpu().numpy())
    predictions = np.array(predictions).argmax(axis = 1)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, predictions

