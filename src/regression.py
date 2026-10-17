from sklearn.linear_model import Ridge
import pandas as pd

def train_ridge_regression(X_train, y_train, alpha: float = 1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def get_model_coefficients(model, feature_names):
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    coef_df = pd.concat([
        pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [model.intercept_]}),
        coef_df
    ], ignore_index=True)
    
    return coef_df