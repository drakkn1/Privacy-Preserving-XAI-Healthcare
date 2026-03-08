import shap
import numpy as np

# -------------------------------------------------------
# Generate SHAP explanations
# -------------------------------------------------------

def generate_shap_explanations(model, X_test):

    # Use a small background sample for SHAP
    background = X_test[:100]

    explainer = shap.KernelExplainer(model.predict, background)

    shap_values = explainer.shap_values(X_test[:100])

    return shap_values


# -------------------------------------------------------
# Explanation filtering (noise + masking)
# -------------------------------------------------------

def add_gaussian_noise(shap_values, sigma=0.05):

    noise = np.random.normal(0, sigma, shap_values.shape)
    return shap_values + noise


def mask_sensitive_features(shap_values, avs_scores, threshold=0.6):

    mask = avs_scores > threshold

    shap_values[:, mask] = 0

    return shap_values