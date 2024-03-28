from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import sys
import pandas as pd


METRICS = ['accuracy', 'precision', 'recall', 'fmeasure', 
           'precision_weighted', 'recall_weighted', 'fmeasure_weighted', 'average_precision_score_weighted',
           'precision_micro', 'recall_micro', 'fmeasure_micro', 'average_precision_score_micro',
           'precision_macro', 'recall_macro', 'fmeasure_macro', 'average_precision_score_macro',
           'roc_auc_score_weighted', 'roc_auc_score_micro', 'roc_auc_score_macro',
           'matthews_corrcoef', 'balanced_accuracy',
           'roc_auc_score_prob_weighted', 'roc_auc_score_prob_micro', 'roc_auc_score_prob_macro']


def compute_metrics(y_test, pred_prob):
    pred = pred_prob > 0.5
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test,pred)

    metrics['precision'] = precision_score(y_test,pred,average='binary')
    metrics['recall'] = recall_score(y_test,pred,average='binary')
    metrics['fmeasure'] = f1_score(y_test,pred,average='binary')

    metrics['precision_weighted'] = precision_score(y_test,pred,average='weighted')
    metrics['recall_weighted'] = recall_score(y_test,pred,average='weighted')
    metrics['fmeasure_weighted'] = f1_score(y_test,pred,average='weighted')
    metrics['average_precision_score_weighted'] = average_precision_score(y_test,pred,average='weighted')

    metrics['precision_micro'] = precision_score(y_test,pred,average='micro')
    metrics['recall_micro'] = recall_score(y_test,pred,average='micro')
    metrics['fmeasure_micro'] = f1_score(y_test,pred,average='micro')
    metrics['average_precision_score_micro'] = average_precision_score(y_test,pred,average='micro')
    
    metrics['precision_macro'] = precision_score(y_test,pred,average='macro')
    metrics['recall_macro'] = recall_score(y_test,pred,average='macro')
    metrics['fmeasure_macro'] = f1_score(y_test,pred,average='macro')
    metrics['average_precision_score_macro'] = average_precision_score(y_test,pred,average='macro')
    
    
    if len(set(y_test)) != 1:
        metrics['roc_auc_score_weighted'] = roc_auc_score(y_test,pred,average='weighted')
        metrics['roc_auc_score_micro'] = roc_auc_score(y_test,pred,average='micro')
        metrics['roc_auc_score_macro'] = roc_auc_score(y_test,pred,average='macro')

    metrics['matthews_corrcoef'] = matthews_corrcoef(y_test,pred) # 0 es random prediction, 1 es perfecto, -1 inverse prediction
    
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test,pred)
    if len(set(y_test)) != 1:
        metrics['roc_auc_score_prob_weighted'] = roc_auc_score(y_test,pred_prob,average='weighted')
        metrics['roc_auc_score_prob_micro'] = roc_auc_score(y_test,pred_prob,average='micro')
        metrics['roc_auc_score_prob_macro'] = roc_auc_score(y_test,pred_prob,average='macro')

    return metrics


def metrics_to_csv(metrics, columns=METRICS, file=sys.stdout):
    if type(metrics) == dict:
        metrics = [metrics]
    df = pd.DataFrame(data=metrics, columns=columns)
    df.to_csv(file)
    pass
    