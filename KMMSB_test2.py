import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import joblib # Added for saving model

# =============================================================================
#  User Selection: Choose the dataset to use ('A', 'B', or 'C')
# =============================================================================
selected_dataset = 'C'  # <--- CHANGE THIS VALUE TO 'A', 'B', or 'C'
# =============================================================================

# Set file path and test split ratio based on selection
if selected_dataset == 'A':
    file_path = 'data(A)2.xlsx'
    test_split_ratio = 0.1212
    print(f"Selected Dataset: A2")
    print(f"File Path: {file_path}")
    print(f"Test Split Ratio: {test_split_ratio}")
elif selected_dataset == 'B':
    file_path = 'data(B)2.xlsx'
    test_split_ratio = 0.14
    print(f"Selected Dataset: B2")
    print(f"File Path: {file_path}")
    print(f"Test Split Ratio: {test_split_ratio}")
elif selected_dataset == 'C':
    file_path = 'data(C)2.xlsx'
    test_split_ratio = 0.117
    print(f"Selected Dataset: C")
    print(f"File Path: {file_path}")
    print(f"Test Split Ratio: {test_split_ratio}")
else:
    raise ValueError(f"Invalid dataset selection: '{selected_dataset}'. Please choose 'A', 'B', or 'C'.")

# Load the dataset
print(f"\nLoading data from {file_path}...")
try:
    data = pd.read_excel(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    # Optionally, you might want to exit the script here
    import sys
    sys.exit(1)

# Separate features (X) and target variable (Y)
print("Separating features and target variable...")
X = data.drop('MMP', axis=1)
X = X.iloc[:, 1:-1]  # Remove the first column, keep the rest as features (Assuming this logic is correct for all datasets)
Y = data['MMP']
print(f"Features shape: {X.shape}, Target shape: {Y.shape}")


# Split the data into training and test sets using the selected ratio
print(f"Splitting data with test_size={test_split_ratio} (shuffle=False)...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_ratio, shuffle=False)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


# Initialize the MinMaxScaler and scale the features
print("Scaling features using MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# Ensure Y_train and Y_test are NumPy arrays for easier indexing later
Y_train_np = Y_train.values if isinstance(Y_train, pd.Series) else Y_train
Y_test_np = Y_test.values if isinstance(Y_test, pd.Series) else Y_test


# --- Rest of your code remains the same ---
# (CombinedModel class definition, base_models_defs, evaluate_model function,
#  param_grids, GridSearch optimization loop, model training, evaluation,
#  plotting, comparison, saving, feature importance analysis)

# --- Make sure the rest of the script uses these variables correctly ---
# For example, the GridSearch and model fitting use X_train_scaled, Y_train_np
# The evaluation uses Y_test_np and predictions made on X_test_scaled

# ... (Your existing code for CombinedModel, base models, evaluation etc. follows here) ...

# Define base models (definitions) - Assuming this part is correct
base_models_defs = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gbr', GradientBoostingRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror')),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('ada', AdaBoostRegressor(random_state=42))
]

# Function to evaluate model performance - Assuming this part is correct
def evaluate_model(Y_true, Y_pred, model_name="Model"):
    mse = mean_squared_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")
    return rmse, mae, r2

# Define parameter grids - Assuming this part is correct
param_grids = {
    'rf': {
        'n_estimators': [60, 80, 100, 120, 140, 160, 180, 200],
        'max_depth': [None, 15, 20],
        'min_samples_split': [2, 3, 4]
    },
    'gbr': {
        'n_estimators': [60, 80, 100, 120, 140, 160, 180, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    },
    'xgb': {
        'n_estimators': [80, 100, 120],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.7, 0.9]
    },
    'dt': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 3, 4]
    },
    'ada': {
        'n_estimators': [80, 100, 120],
        'learning_rate': [0.01, 0.1, 1.0]
    }
}

# CombinedModel Class Definition (assuming it's already defined above)
class CombinedModel(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, blending_estimator=LinearRegression(), n_splits=5, random_state=42):
        self.base_models_defs = base_models # Store model definitions
        self.base_models_trained = [] # To store models trained on full data
        self.blending_model = clone(blending_estimator) # Use clone to avoid modifying the original
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X_train_scaled, Y_train):
        # Ensure Y_train is a NumPy array
        Y_train = Y_train.values if isinstance(Y_train, pd.Series) else Y_train

        num_train = X_train_scaled.shape[0]
        num_models = len(self.base_models_defs)

        # Array to store out-of-fold predictions for training the blending model
        oof_predictions = np.zeros((num_train, num_models))

        # Use KFold for cross-validation to generate OOF predictions
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        print(f"Generating OOF predictions using {self.n_splits}-Fold CV...")
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_scaled, Y_train)):
            # print(f"  Processing Fold {fold_idx+1}/{self.n_splits}")
            X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
            Y_train_fold = Y_train[train_index] # Corrected: Get Y_train_fold

            # Train each base model on the training fold and predict on the validation fold
            for i, (name, model_def) in enumerate(self.base_models_defs):
                # Clone the model definition for independent training in each fold
                model = clone(model_def)
                model.fit(X_train_fold, Y_train_fold)
                # Store the predictions made on the validation fold
                oof_predictions[val_index, i] = model.predict(X_val_fold)

        # Train the blending model using the OOF predictions as features and original Y_train as target
        print("Training blending model on OOF predictions...")
        self.blending_model.fit(oof_predictions, Y_train)

        # --- Crucial Step: Retrain base models on the *entire* training set ---
        # These retrained models will be used in the predict method
        print("Retraining base models on the full training set...")
        self.base_models_trained = [] # Clear previous list if any
        for name, model_def in self.base_models_defs:
            model = clone(model_def) # Clone again
            model.fit(X_train_scaled, Y_train)
            self.base_models_trained.append((name, model)) # Store the trained model
        print("Fit process complete.")
        return self

    def predict(self, X_scaled):
        # Generate base model predictions using models trained on the full training set
        predictions_scaled = np.zeros((X_scaled.shape[0], len(self.base_models_trained)))
        for i, (_, model) in enumerate(self.base_models_trained):
            predictions_scaled[:, i] = model.predict(X_scaled)

        # Use the trained blending model to make final predictions
        blending_predictions = self.blending_model.predict(predictions_scaled)
        return blending_predictions


# Optimize base models - Assuming this part is correct
optimized_base_models = []
print("\nStarting grid search to optimize base models...")
for name, model in base_models_defs:
    print(f"Optimizing {name} model...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, Y_train_np)
    best_model = grid_search.best_estimator_
    optimized_base_models.append((name, best_model))
    print(f"{name} best parameters: {grid_search.best_params_}")
    print(f"{name} best performance (neg MSE): {grid_search.best_score_:.4f}")
    print("-" * 50)

# Train and evaluate original and optimized models - Assuming this part is correct
print("\nTraining combined model using original unoptimized models...")
original_combined_model = CombinedModel(
    base_models=base_models_defs,
    blending_estimator=LinearRegression(),
    n_splits=5,
    random_state=42
)
original_combined_model.fit(X_train_scaled, Y_train_np)
original_predictions = original_combined_model.predict(X_test_scaled)
print("\nEvaluating original combined model:")
original_rmse, original_mae, original_r2 = evaluate_model(Y_test_np, original_predictions, "Original Combined Model")

print("\nTraining combined model using optimized base models...")
optimized_combined_model = CombinedModel(
    base_models=optimized_base_models,
    blending_estimator=LinearRegression(),
    n_splits=5,
    random_state=42
)
optimized_combined_model.fit(X_train_scaled, Y_train_np)
optimized_predictions = optimized_combined_model.predict(X_test_scaled)
print("\nEvaluating optimized combined model:")
optimized_rmse, optimized_mae, optimized_r2 = evaluate_model(Y_test_np, optimized_predictions, "Optimized Combined Model")

# Plotting - Assuming this part is correct
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(original_predictions, Y_test_np, color='blue', alpha=0.6, label='Predicted')
min_val_orig = min(Y_test_np.min(), original_predictions.min())
max_val_orig = max(Y_test_np.max(), original_predictions.max())
plt.plot([min_val_orig, max_val_orig], [min_val_orig, max_val_orig], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Predicted vs Actual (Original Combined Model)')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(optimized_predictions, Y_test_np, color='green', alpha=0.6, label='Predicted')
min_val_opt = min(Y_test_np.min(), optimized_predictions.min())
max_val_opt = max(Y_test_np.max(), optimized_predictions.max())
plt.plot([min_val_opt, max_val_opt], [min_val_opt, max_val_opt], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Predicted vs Actual (Optimized Combined Model)')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# Comparison - Assuming this part is correct
print("\nModel Performance Comparison:")
print("-" * 60)
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
print(f"{'Original Model':<20} {original_rmse:<10.4f} {original_mae:<10.4f} {original_r2:<10.4f}")
print(f"{'Optimized Model':<20} {optimized_rmse:<10.4f} {optimized_mae:<10.4f} {optimized_r2:<10.4f}")

rmse_improvement = (original_rmse - optimized_rmse) / original_rmse * 100 if original_rmse != 0 else float('inf')
mae_improvement = (original_mae - optimized_mae) / original_mae * 100 if original_mae != 0 else float('inf')
r2_improvement = (optimized_r2 - original_r2) / abs(original_r2) * 100 if original_r2 != 0 else float('inf')

print(f"{'Improvement %':<20} {rmse_improvement:<10.2f}% {mae_improvement:<10.2f}% {r2_improvement:<10.2f}%")
print("-" * 60)

# Save optimized model - Assuming this part is correct
model_filename = f'optimized_combined_model_{selected_dataset}.pkl'
joblib.dump(optimized_combined_model, model_filename)
print(f"\nOptimized model saved as '{model_filename}'")

# Feature Importance Analysis - Assuming this part is correct
print("\nBase Model Feature Importance:")
if X.columns is not None:
    feature_names = X.columns
    for name, model in optimized_base_models:
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name} model feature importance:")
            importances = model.feature_importances_
            # Ensure feature_names length matches importances length
            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                print(importance_df.head(10))

                plt.figure(figsize=(10, 6))
                plt.barh(importance_df['Feature'].head(10)[::-1], importance_df['Importance'].head(10)[::-1])
                plt.xlabel('Importance')
                plt.title(f'{name} Model ({selected_dataset} Dataset) - Top 10 Important Features')
                plt.tight_layout()
                plt.show()
            else:
                 print(f"Warning: Feature count for {name} model ({len(importances)}) doesn't match column count in X ({len(feature_names)}). Skipping feature importance plot.")
        else:
            print(f"{name} model doesn't support feature_importances_ attribute.")
else:
    print("Warning: Cannot retrieve feature names (X.columns is None). Skipping feature importance analysis.")

print("\nScript execution completed.")
