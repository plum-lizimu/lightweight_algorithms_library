from task_embarkation.cifar_task.cifarEmbarkation import cifarKDEmbarkation, cifarPruningEmbarkation, cifarQATEmbarkation, cifarDynamicQuantizationEmbarkation
from task_embarkation.agNews_task.agNewsEmbarkation import agNewsEmbarkation

embarkation_dict = {
    'cifarKDEmbarkation': cifarKDEmbarkation,
    'cifarPruningEmbarkation': cifarPruningEmbarkation,
    'cifarQATEmbarkation': cifarQATEmbarkation,
    'cifarDynamicQuantizationEmbarkation': cifarDynamicQuantizationEmbarkation,
    'agNewsEmbarkation': agNewsEmbarkation 
}