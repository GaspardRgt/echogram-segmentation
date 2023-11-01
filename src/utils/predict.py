import torch
from torch import nn
import torch.nn.functional as F

# Detailed predictions on multiple layers

def detailed_predictions(model, X):
    softmax = nn.Softmax(dim=1)
    preds_dict = {}
    model.eval()
    with torch.inference_mode():
        enc_features = model.encoder(X)[::-1]
        dec_features = model.decoder(enc_features[0], enc_features[1:])
        masks_logits = model.head(dec_features)
        if model.retain_dim:
            masks_logits = F.interpolate(masks_logits, model.out_size)

        masks_pred_probs = softmax(masks_logits)
        pred_labels = torch.argmax(masks_pred_probs, dim=1)

    preds_dict['enc_features'] = [x.detach() for x in enc_features]
    preds_dict['dec_features'] = dec_features.detach()
    preds_dict['masks_logits'] = masks_logits.detach()
    preds_dict['masks_pred_probs'] = masks_pred_probs.detach()
    preds_dict['pred_labels'] = pred_labels.detach()

    return preds_dict