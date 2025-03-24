import os
import math
import torch
import torchmetrics
import numpy as np
import random
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def seed_all(seed):
    if not seed:
        seed = 0
    print("Using Seed : ", seed)

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# EARLY STOPPING ON VALIDATION ACCURACY
class EarlyStopping_:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, path='', trace_func=print, warm_up_epochs=0):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'model_trained_10.pt'
            trace_func (function): trace print function.
                            Default: print
            warm_up_epochs (int): Number of epochs to run before starting early stopping checks.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_accuracy = -np.Inf
        self.path = path
        self.warm_up_epochs = warm_up_epochs
        self.trace_func = trace_func
        self.epoch_count = 0

    def __call__(self, val_accuracy, model):
        self.epoch_count += 1
        if self.epoch_count <= self.warm_up_epochs:
            return

        if val_accuracy > self.best_accuracy:
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_accuracy, model):
        '''Saves model when validation accuracy increases.'''
        if self.verbose:
            self.trace_func(
                f'Validation accuracy increased ({self.best_accuracy:.6f} --> {val_accuracy:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_accuracy = val_accuracy


# EARLY STOPPING ON VALIDATION LOSS
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, path='', trace_func=print, warm_up_epochs=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'model_trained_10.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_loss = np.Inf
        self.path = path
        self.warm_up_epochs = warm_up_epochs
        self.trace_func = trace_func
        self.epoch_count = 0

    def __call__(self, val_loss, model):

        self.epoch_count += 1
        if self.epoch_count <= self.warm_up_epochs:
            return

        if val_loss < self.best_loss:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

        self.best_loss = val_loss


##########################################################################################
# Training with early stopping on validation with sigmoid activation and optimal threshold
def Training_v2(K_fold, n_epochs, model, train_loader, valid_loader, optimizer, scheduler, criterion, early_stopping, History, device):
    valid_accuracy = torchmetrics.Accuracy("binary").to(device)
    all_train_outputs = []
    all_train_labels = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        Loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(images.float()).squeeze(dim=1)
                # print(outputs, torch.sigmoid(outputs))
                loss = criterion(torch.sigmoid(outputs), labels.type(torch.cuda.FloatTensor))
                probabilities = torch.sigmoid(outputs)
                all_train_outputs.extend(probabilities.detach().cpu().numpy())
                all_train_labels.extend(labels.detach().cpu().numpy())

                loss.backward()  # retain_graph=True
                optimizer.step()

            Loss += loss.item() * images.size(0)

        Loss = Loss / len(train_loader)  # This calculates the average loss per batch for the epoch. Since the loss
        # for each batch was an average loss per data point multiplied by the number of data points, dividing
        # the total loss by the number of batches gives you the average loss per data point over the entire epoch.

        History['loss_Function'][str(K_fold)]['Training'].append(Loss)

        print('Fold: {} \tEpoch: {}/{} \nTraining Loss: {:.4f}'.format(K_fold, epoch, n_epochs, Loss))

        # ----------------------------- VALIDATION ------------------------------------- #
        model.eval()

        # Compute predictions using the optimal threshold on the training set
        train_predictions = (np.array(all_train_outputs) > 0.5).astype(int)

        # Compute train accuracy with optimal threshold
        train_acc = accuracy_score(all_train_labels, train_predictions)
        print(f"Training Accuracy with optimal threshold: {train_acc.item()}")

        Loss = 0.0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(images.float()).squeeze(dim=1)
                loss = criterion(torch.sigmoid(outputs), labels.type(torch.cuda.FloatTensor))
                probabilities = torch.sigmoid(outputs)
                preds = (probabilities > 0.5)

            Loss += loss.item() * images.size(0)
            valid_accuracy.update(preds, labels.type(torch.cuda.LongTensor))
        Loss = Loss / len(valid_loader)

        History['loss_Function'][str(K_fold)]['Validation'].append(Loss)

        avg_val_acc = valid_accuracy.compute().item()
        History['Accuracy_x_fold'][str(K_fold)]['Validation'].append(avg_val_acc)

        print('Validation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(Loss, avg_val_acc))

        scheduler.step(Loss)

        History['Accuracy_x_fold'][str(K_fold)]['Training'].append(train_acc.item())
        History['Learning_Rate'][str(K_fold)].append(optimizer.param_groups[0]['lr'])
        print('Actual LR: {}'.format(optimizer.param_groups[0]['lr']))

        early_stopping(avg_val_acc, model)  # if early stopping on validation loss use Loss
        torch.cuda.empty_cache()
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training Fold: {}'.format(K_fold))
    print("-" * 50)

    return History


# ----------------------------------------------------------------------------------------
# test function for one-leave-one-out patient with best threshold sigmoid (whole lungs)
def Testing_v4(K_fold, model, test_loader, History_test, Metrics, path_results, device):
    print("-" * 80)
    print('Testing fold {}'.format(K_fold))
    all_outputs = []
    all_labels = []
    patient_ids = []

    model.eval()

    with torch.no_grad():
        for images, labels, patient_id in test_loader:
            # break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.float())
            probabilities = torch.sigmoid(outputs)

            all_outputs.extend(probabilities.detach().cpu().numpy()[0])
            all_labels.extend(labels.detach().cpu().numpy())
            patient_ids.extend(patient_id)

        df1 = pd.DataFrame({'subjectid': patient_ids, 'label': all_labels, 'probability output': all_outputs})
        df1.to_csv(f"{path_results}/outputs_{K_fold}.csv")

        # Get the unique patient IDs
        unique_patient_ids = df1['subjectid']
        print(unique_patient_ids)
        # Iterate over the unique patient IDs
        prob_outputs = []
        labels = []
        thrs = []
        predictions = []
        ids = []

        for p_id in unique_patient_ids:

            roc = {"th": [], "FPR": [], "TPR": []}
            leave_out_patient = df1[df1['subjectid'] == p_id]
            other_patients = df1[~(df1['subjectid'] == p_id)]

            other_patients_labels = other_patients['label'].tolist()
            other_patients_prob = other_patients['probability output'].tolist()

            fpr, tpr, thresholds = roc_curve(other_patients_labels, other_patients_prob)

            roc['FPR'].extend(fpr)
            roc['TPR'].extend(tpr)
            roc['th'].extend(thresholds)
            roc = json.dumps(roc)

            with open(f'{path_results}/roc_val_{K_fold}_without_pat_{p_id}.json', 'w') as f:
                f.write(roc)

            distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)  # calculate distances from (0,1)
            optimal_idx = np.argmin(distances)  # find index of the smallest distance
            optimal_threshold = thresholds[optimal_idx]

            # Compute predictions using the optimal threshold
            for subjectid in leave_out_patient["subjectid"]:
                leave_out_patient_prob = leave_out_patient[leave_out_patient["subjectid"] == subjectid][
                    "probability output"]
                preds = (np.array(leave_out_patient_prob) > optimal_threshold).astype(int)  # optimal_threshold

                prob_outputs.append(leave_out_patient_prob.values[0])
                ids.append(subjectid)
                predictions.append(preds[0])
                thrs.append(optimal_threshold)
                labels.append(leave_out_patient[leave_out_patient["subjectid"] == subjectid]["label"].values[0])

        df2 = pd.DataFrame({'subjectid': ids, 'label': labels, 'probability output': prob_outputs, 'thresholds': thrs,
                            'predictions': predictions})
        df2.to_csv(f"{path_results}/output_predictions_{K_fold}.csv")

        # compute the performance
        for pred, label in zip(df2["predictions"], df2["label"]):
            # EVALUATION
            if pred != label:
                if label == 0:
                    History_test['CM'][str(K_fold)]['FP'] += 1
                else:
                    History_test['CM'][str(K_fold)]['FN'] += 1
            else:
                if label == 0:
                    History_test['CM'][str(K_fold)]['TN'] += 1
                else:
                    History_test['CM'][str(K_fold)]['TP'] += 1

        Metrics['Recall'][str(K_fold)] = History_test['CM'][str(K_fold)]['TP'] \
                                         / (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'FN'])

        Metrics['Precision'][str(K_fold)] = History_test['CM'][str(K_fold)]['TP'] \
                                            / (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'FP'])

        Metrics['Specificity'][str(K_fold)] = History_test['CM'][str(K_fold)]['TN'] \
                                              / (History_test['CM'][str(K_fold)]['TN'] +
                                                 History_test['CM'][str(K_fold)]['FP'])

        Metrics['F1_score'][str(K_fold)] = (2 * Metrics['Precision'][str(K_fold)] * Metrics['Recall'][str(K_fold)]) \
                                           / (Metrics['Precision'][str(K_fold)] + Metrics['Recall'][str(K_fold)])

        Metrics['G_mean'][str(K_fold)] = math.sqrt(Metrics['Recall'][str(K_fold)] * Metrics['Specificity'][str(K_fold)])

        Metrics['Accuracy'][str(K_fold)] = (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'TN']) \
                                           / (History_test['CM'][str(K_fold)]['TP'] +
                                              History_test['CM'][str(K_fold)]['TN'] +
                                              History_test['CM'][str(K_fold)]['FP'] +
                                              History_test['CM'][str(K_fold)]['FN'])

    return History_test, Metrics


def Testing_tSNE_All_Folds(model, p, model_paths, test_loaders, path_results, device):
    """
    Perform t-SNE visualization using embeddings from all folds.

    Args:
        model_paths (list): List of file paths for trained models (one per fold).
        test_loaders (list): List of DataLoaders for each fold.
        path_results (str): Path to save the t-SNE visualization.
        device (torch.device): Device (CPU/GPU).
    """

    print("-" * 80)
    # print(f'Extracting embeddings from all {len(model_paths)} folds for t-SNE visualization')

    all_embeddings = []
    all_labels = []

    for fold_idx, (model_path, test_loader) in enumerate(zip(model_paths, test_loaders)):
        print(f"Processing Fold {fold_idx + 1}...")

        # Load model for current fold
        state_dict = torch.load(model_path, map_location=device)
        # Remove "module." prefix if exists
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value

        # Load the modified state_dict into the model
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()

        fold_embeddings = []
        fold_labels = []

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Extract embeddings
                embeddings = model(images.float(), return_embeddings=True)
                print(embeddings)
                # Store embeddings and labels
                fold_embeddings.append(embeddings.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())

        # Concatenate embeddings for this fold
        fold_embeddings = np.vstack(fold_embeddings)
        fold_labels = np.array(fold_labels)

        print(f"Fold {fold_idx + 1}: Extracted {fold_embeddings.shape[0]} embeddings of size {fold_embeddings.shape[1]}")

        # Append to global storage
        all_embeddings.append(fold_embeddings)
        all_labels.append(fold_labels)

    # Concatenate all folds
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    print(f"Total Combined Embeddings: {all_embeddings.shape[0]}, Dimension: {all_embeddings.shape[1]}")

    # Save embeddings and labels for later visualization
    os.makedirs(path_results, exist_ok=True)
    embeddings_path = os.path.join(path_results, "tSNE_Embeddings.npz")

    np.savez_compressed(embeddings_path, embeddings=all_embeddings, labels=all_labels)

    print(f"t-SNE embeddings saved at: {embeddings_path}")

    # Perform t-SNE on the combined embeddings
    tsne = TSNE(n_components=2, perplexity=p, learning_rate=200, n_iter=5000, init='pca', metric='euclidean', random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    print(all_labels.shape, embeddings_2d.shape)
	
    silhouette = silhouette_score(embeddings_2d, all_labels)
	
    print(f"Perplexity: {p}, Silhouette Score: {silhouette}")
	
    # Define custom colors: 0 -> Green, 1 -> Red
    custom_palette = {0: "green", 1: "red"}
    
    # Plot t-SNE
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=all_labels, palette=custom_palette, alpha=0.7, s=220)
    # plt.title("t-SNE Visualization of Combined Fold Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # plt.legend(title="Class Labels", loc="best")

    # Remove axis labels
    plt.xlabel("")
    plt.ylabel("")

    # Remove ticks and tick labels
    plt.xticks([])
    plt.yticks([])

    # Remove axis spines (borders)
    #for spine in plt.gca().spines.values():
    #    spine.set_visible(False)

    # Save the plot
    tsne_path = f"{path_results}/tSNE_All_Folds_perplexity_{p}.pdf"
    plt.savefig(tsne_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"t-SNE visualization saved at: {tsne_path}")


# get the prediction on the training set with thresold 0.5 (whole lungs)
def Testing_v5(K_fold, model, test_loader, History_test, Metrics, path_results, device):
    print("-" * 80)
    print('Testing fold {}'.format(K_fold))
    all_outputs = []
    all_labels = []
    patient_ids = []
    embeddings = []
    predictions = []
    model.eval()

    with torch.no_grad():
        for images, labels, patient_id in test_loader:
            # break
            images = images.to(device)
            labels = labels.to(device)
            outputs, embedding = model(images.float())
            print(embedding.shape)
            probabilities = torch.sigmoid(outputs)
            preds = (np.array(probabilities.detach().cpu().numpy()[0]) > 0.5).astype(int)

            all_outputs.extend(probabilities.detach().cpu().numpy()[0])
            embeddings.extend(embedding.detach().cpu().numpy())
            predictions.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            patient_ids.extend(patient_id)

        df1 = pd.DataFrame({'subjectid': patient_ids, 'predictions': predictions, 'label': all_labels, 'probability output': all_outputs})
        # df1.to_csv(f"{path_results}/outputs_training_set_{K_fold}.csv")
        embeddings_array = np.array(embeddings)
        #  np.save(f"{path_results}/embeddings_traininf_set_{K_fold}.npy", embeddings_array)

        # compute the performance
        for pred, label in zip(df1["predictions"], df1["label"]):
            # EVALUATION
            if pred != label:
                if label == 0:
                    History_test['CM'][str(K_fold)]['FP'] += 1
                else:
                    History_test['CM'][str(K_fold)]['FN'] += 1
            else:
                if label == 0:
                    History_test['CM'][str(K_fold)]['TN'] += 1
                else:
                    History_test['CM'][str(K_fold)]['TP'] += 1

        Metrics['Recall'][str(K_fold)] = History_test['CM'][str(K_fold)]['TP'] \
                                         / (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'FN'])

        Metrics['Precision'][str(K_fold)] = History_test['CM'][str(K_fold)]['TP'] \
                                            / (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'FP'])

        Metrics['Specificity'][str(K_fold)] = History_test['CM'][str(K_fold)]['TN'] \
                                              / (History_test['CM'][str(K_fold)]['TN'] +
                                                 History_test['CM'][str(K_fold)]['FP'])

        Metrics['F1_score'][str(K_fold)] = (2 * Metrics['Precision'][str(K_fold)] * Metrics['Recall'][str(K_fold)]) \
                                           / (Metrics['Precision'][str(K_fold)] + Metrics['Recall'][str(K_fold)])

        Metrics['G_mean'][str(K_fold)] = math.sqrt(Metrics['Recall'][str(K_fold)] * Metrics['Specificity'][str(K_fold)])

        Metrics['Accuracy'][str(K_fold)] = (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'TN']) \
                                           / (History_test['CM'][str(K_fold)]['TP'] +
                                              History_test['CM'][str(K_fold)]['TN'] +
                                              History_test['CM'][str(K_fold)]['FP'] +
                                              History_test['CM'][str(K_fold)]['FN'])

    return History_test, Metrics


def tta(K_fold, model, test_loader, History_test, Metrics, path_results, device):
    # campione più vicino sul training set
    def predict_label(test_embedding, train_embeddings, train_labels):
        # Compute distances (e.g., Euclidean)
        distances = cdist(test_embedding.reshape(1, -1), train_embeddings, metric='euclidean')
        closest_idx = np.argmin(distances)  # Index of the closest training embedding
        return train_labels[closest_idx]

    def predict_label_knn(test_embedding, train_embeddings, train_labels, k=3):
        # Compute distances
        distances = cdist(test_embedding.reshape(1, -1), train_embeddings, metric='euclidean').flatten()
        # Get indices of the K closest embeddings
        k_closest_idx = np.argsort(distances)[:k]
        # Get the labels of the K closest embeddings
        k_closest_labels = train_labels[k_closest_idx]
        # Majority vote
        most_common = Counter(k_closest_labels).most_common(1)
        return most_common[0][0]

    print("-" * 80)
    print('Testing fold {}'.format(K_fold))
    all_outputs = []
    all_labels = []
    patient_ids = []
    embeddings = []
    predictions = []
    model.eval()
    training_outputs = pd.read_csv(f"{path_results}/outputs_training_set_{K_fold}.csv")
    train_labels = training_outputs["label"]
    train_embeddings = np.load(f"{path_results}/embeddings_traininf_set_{K_fold}.npy")

    with torch.no_grad():
        for images, labels, patient_id in test_loader:
            # break
            images = images.to(device)
            labels = labels.to(device)
            _, test_embedding = model(images.float())

            test_prediction = predict_label_knn(test_embedding.detach().cpu().numpy(), train_embeddings, train_labels, k=10)
            # test_prediction = predict_label(test_embedding.detach().cpu().numpy(), train_embeddings, train_labels)
            embeddings.extend(test_embedding.detach().cpu().numpy())
            predictions.extend(np.array([test_prediction]))
            all_labels.extend(labels.detach().cpu().numpy())
            patient_ids.extend(patient_id)

        df1 = pd.DataFrame({'subjectid': patient_ids, 'predictions': predictions, 'label': all_labels})
        df1.to_csv(f"{path_results}/outputs_test_set_{K_fold}_knn_10.csv")
        embeddings_array = np.array(embeddings)
        np.save(f"{path_results}/embeddings_test_set_{K_fold}_knn_10.npy", embeddings_array)

        # compute the performance
        for pred, label in zip(df1["predictions"], df1["label"]):
            # EVALUATION
            if pred != label:
                if label == 0:
                    History_test['CM'][str(K_fold)]['FP'] += 1
                else:
                    History_test['CM'][str(K_fold)]['FN'] += 1
            else:
                if label == 0:
                    History_test['CM'][str(K_fold)]['TN'] += 1
                else:
                    History_test['CM'][str(K_fold)]['TP'] += 1

        Metrics['Recall'][str(K_fold)] = History_test['CM'][str(K_fold)]['TP'] \
                                         / (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'FN'])

        Metrics['Precision'][str(K_fold)] = History_test['CM'][str(K_fold)]['TP'] \
                                            / (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'FP'])

        Metrics['Specificity'][str(K_fold)] = History_test['CM'][str(K_fold)]['TN'] \
                                              / (History_test['CM'][str(K_fold)]['TN'] +
                                                 History_test['CM'][str(K_fold)]['FP'])

        Metrics['F1_score'][str(K_fold)] = (2 * Metrics['Precision'][str(K_fold)] * Metrics['Recall'][str(K_fold)]) \
                                           / (Metrics['Precision'][str(K_fold)] + Metrics['Recall'][str(K_fold)])

        Metrics['G_mean'][str(K_fold)] = math.sqrt(Metrics['Recall'][str(K_fold)] * Metrics['Specificity'][str(K_fold)])

        Metrics['Accuracy'][str(K_fold)] = (History_test['CM'][str(K_fold)]['TP'] + History_test['CM'][str(K_fold)][
            'TN']) \
                                           / (History_test['CM'][str(K_fold)]['TP'] +
                                              History_test['CM'][str(K_fold)]['TN'] +
                                              History_test['CM'][str(K_fold)]['FP'] +
                                              History_test['CM'][str(K_fold)]['FN'])

    return History_test, Metrics


# self-supervised pretraining script
def pre_training(n_epochs, train_loader, optimizer, scheduler, b_twins, Tracking, path_model, device):
    # Loop over the epochs
    for epoch in range(n_epochs):
        # Set the model to training mode
        b_twins.train()

        # Loop over the batches in the train loader
        for i, (y1, y2) in tqdm(enumerate(train_loader)):
            # Move the images to the device
            y1 = y1.to(device)
            y2 = y2.to(device)

            # Forward pass
            loss = b_twins(y1, y2)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the scheduler
            scheduler.step(loss)

            # Save the loss for tracking
            Tracking['loss_Function'].append(loss.item())

        print(f"Epoch: {epoch}/{n_epochs}, B_twins Loss:{loss.item()}")
        # Save the model after each epoch
        torch.save(b_twins.state_dict(), path_model)

    return Tracking


def pre_training_grad_accumulation(n_epochs, train_loader, optimizer, scheduler, b_twins, Tracking, path_model, device):
    accumulation_steps = 1  # 16 batch simulated with 4 batch per step

    for epoch in range(n_epochs):
        b_twins.train()
        running_loss = 0.0  # Track loss

        for i, (y1, y2) in tqdm(enumerate(train_loader), total=len(train_loader)):
            y1, y2 = y1.to(device), y2.to(device)

            # Forward pass
            loss = b_twins(y1, y2) / accumulation_steps  # Scale loss

            # Backward pass (no optimizer step yet)
            loss.backward()

            # Accumulate gradients and update only after 'accumulation_steps' batches
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()  # Update model weights
                optimizer.zero_grad()  # Reset gradients
                scheduler.step(loss * accumulation_steps)  # Rescale loss before stepping

            running_loss += loss.item() * accumulation_steps  # Track actual loss

        # Save loss for tracking
        Tracking['loss_Function'].append(running_loss / len(train_loader))
        print(f"Epoch: {epoch + 1}/{n_epochs}, B_twins Loss: {running_loss / len(train_loader):.6f}")

        # Save model after each epoch
        torch.save(b_twins.state_dict(), path_model)

    return Tracking
