from keras.models import load_model
from utils import encode_batch, scale_output
import numpy as np

def predict_scores(gRNA_list):
    model = load_model("model/gRNA_uv_model.h5")
    encoded = encode_batch(gRNA_list)
    preds = model.predict(encoded).flatten()
    scaled = scale_output(preds)
    
    return dict(zip(gRNA_list, scaled))




if __name__ == "__main__":
    gRNA_list = []
    scores = predict_scores(gRNA_list)
    print(scores)
