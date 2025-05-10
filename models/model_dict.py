from models.cv_model.resnet import resnet34, resnet18
from models.rs_model.DNN import DNN
from models.rs_model.DeepFM import DeepFM
from models.rs_model.DCN import DCN
from transformers import BertModel, BertForSequenceClassification, AutoModelForSequenceClassification

model_dict = {
    # ResNet models
    'resnet18': resnet18,
    'resnet34': resnet34,

    # CTR models
    'DNN': DNN,
    'DeepFM': DeepFM,
    'DCN': DCN,

    # NLP models (在线)
    'bert': BertModel,
    'bert_for_sequence_classification': BertForSequenceClassification,
    'auto_model_for_sequence_classification': AutoModelForSequenceClassification,

    # Add other models here
}