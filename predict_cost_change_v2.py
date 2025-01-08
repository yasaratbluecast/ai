import pandas as pd
import numpy as np
import warnings

# Suppress specific UserWarnings
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer*",
    category=UserWarning,
)
warnings.simplefilter("ignore", category=UserWarning)

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def fix_date(row):
    """Helper function to safely construct the datetime string."""
    try:
        year = str(row['year'])
        month = str(row['month'])
        return pd.to_datetime(f"{year}-{month}-01")  # Add a default day (1st)
    except (KeyError, TypeError):
        return pd.NaT  # Or handle the error as needed

def read_config_to_dict(file_path):
    config_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if '=' in line:
                    key, value = map(str.strip, line.split('=', 1))
                    try:
                        config_dict[key] = float(value)  # Use float instead of int
                    except ValueError:
                        print(f"Error: Could not convert value '{value}' for key '{key}' to a float.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return config_dict

# Predicator scaling
file_path = "config/predictors_config.cfg"
config = read_config_to_dict(file_path)
print(config)

def process_ingredients(file_path):
    """
    Reads a CSV file, extracts specific columns, and cleans the data.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Processed DataFrame with cleaned 'Ingredient' and 'Usage' columns.
    """
    res = pd.DataFrame()
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Extract only the first two columns
    df = df.iloc[:, :3]  # Select the first two columns: 'Ingredient' and 'Usage'
    # Rename columns for clarity
    df.columns = ['Ingredient', 'Usage','Series_ID']
    # Remove spaces from 'Ingredient' column
    res['Ingre'] = df['Ingredient'].str.replace(" ", "") +"_" + df['Series_ID']
    # Remove '%' from 'Usage' column and convert to numeric
    res['Usage'] = df['Usage'].str.replace("%", "").astype(float)
    return res

# Example usage
ingre_file_path = "data/Pizza_Ingredients_matched.csv"
ingre_df = process_ingredients(ingre_file_path)

def read_msa_data(file_path):
    # df_msa = pd.DataFrame()
    df = pd.read_csv(file_path)
    df['year'] = df['year'].astype(str)
    df['month'] = df['month'].astype(str)
    df['date'] = df.apply(fix_date, axis=1)
    df.drop(columns=['year', 'month','index'],inplace=True)
    df.columns = df.columns.str.replace(" ", "")
    df_msa = df.interpolate(method='linear')
    df_msa.set_index('date', inplace=True)

    # Iterate through the Ingre DataFrame
    for index, row in ingre_df.iterrows():
        ingredient = row['Ingre']  # Get the column name from 'Ingre'
        usage_value = row['Usage']  # Get the corresponding 'Usage' value
        # Apply the usage value to the matching column in the data DataFrame
        if ingredient in df_msa.columns:
            df_msa[ingredient] = df_msa[ingredient] * usage_value
        else:
            print(f"Warning: {ingredient} not found in data DataFrame columns.")

    df_msa['fed_int_rate'] = df_msa['fed_int_rate'] * 1000
    df_msa['us_umemploy_r'] = df_msa['us_umemploy_r'] * 1000
    df_msa['electricity_kwh'] = df_msa['electricity_kwh'] * 10000
    df_msa['reg_gasoline'] = df_msa['reg_gasoline'] * 1000
    df_msa['fnb_shops_emp'] = df_msa['fnb_shops_emp'] * 10
    df_msa['home_price_idx'] = df_msa['home_price_idx'] * 10

    return df_msa

atl_piz_file = "data/Atlanta-pizzeria_micro_econ.csv"
dal_piz_file = "data/Dallas-pizzeria_micro_econ.csv"
lax_piz_file = "data/LosAngeles-pizzeria_micro_econ.csv"

atl_data = read_msa_data(atl_piz_file)
dal_data = read_msa_data(dal_piz_file)
lax_data = read_msa_data(lax_piz_file)

# Select only 'cpi_food_away_from_home' column
target_column = 'cpi_food_away_from_home'
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Load the saved model
atl_model_hy_cnn = load_model('model/atl_model_hy_cnn_v2.keras')
lax_model_hy_cnn = load_model('model/lax_model_hy_cnn_v2.keras')
dal_model_hy_cnn = load_model('model/dal_model_hy_cnn_v2.keras')
atl_base = 245.09427
dal_base = 252.8825
lax_base = 268.20584


def update_next_x_vals(next_x_vals, config, column_names):
    """
    Updates the columns of next_x_vals based on the config dictionary.
    Args:
        next_x_vals (numpy.ndarray): Array representing the attributes to be updated.
        config (dict): Dictionary with keys as attribute names and values as multipliers.
        column_names (list): List of column names corresponding to the columns in next_x_vals.
    Returns:
        numpy.ndarray: Updated next_x_vals array.
    """
    for idx, col_name in enumerate(column_names):
        multiplier = config.get(col_name, 1.0)  # Default multiplier is 1.0 if the key is missing
        if multiplier is not None:
            next_x_vals[:, idx] = next_x_vals[:, idx] * multiplier
        else:
            print(f"Warning: No multiplier found for {col_name}, using default 1.0.")
    return next_x_vals

def predict_next_value(model_hy_cnn, next_x_vals, base):
    # Assuming SEQ_LEN is the sequence length the model was trained on
    # next_x_vals has shape (1, 19), so we need to reshape it appropriately
    SEQ_LEN = 12  # Replace with your actual SEQ_LEN
    # Reshape next_x_vals to (1, 1, 19) before repeating
    next_x_vals_reshaped = next_x_vals.reshape(1, 1, 19)
    # Repeat along axis=1 to get (1, SEQ_LEN, 19)
    next_x_vals_repeated = np.repeat(next_x_vals_reshaped, SEQ_LEN, axis=1)
    # Scale the input using the scaler fitted on the training data
    # Assuming scaler_x is the scaler used for input features
    next_x_vals_scaled = scaler_x.transform(next_x_vals_repeated.reshape(-1, 19))
    next_x_vals_scaled = next_x_vals_scaled.reshape(1, SEQ_LEN, 19)  # Reshape back to (1, SEQ_LEN, 19)
    # Predict the target variable using the scaled input
    next_y_pred_scaled = model_hy_cnn.predict(next_x_vals_scaled)
    # If the target variable was scaled, inverse transform the prediction
    next_y_pred = scaler_y.inverse_transform(next_y_pred_scaled)
    res_y = next_y_pred[0][0]
    y_growth_pct = np.round(100 * (res_y - base) / base, 2)

    return np.round(res_y,2), y_growth_pct

atl_val_y = atl_data[[target_column]].values
atl_val_x = atl_data.drop(target_column, axis=1).values
# Scale the data
scaled_val_x = scaler_x.fit_transform(atl_val_x)
scaled_val_y = scaler_y.fit_transform(atl_val_y)
last_x_vals = atl_val_x[-1:]
next_x_vals = last_x_vals.copy()
# print("ATL nextx: ", next_x_vals)
updated_next_x_vals = update_next_x_vals(next_x_vals, config, config.keys())

atl_y_val, atl_growth = predict_next_value(atl_model_hy_cnn, updated_next_x_vals, atl_base)
# Display the predicted response variable
print(f"Atlanta --> Response variable value for the given changes in the Predictor values:{np.round(atl_y_val,2)} with base value:{np.round(atl_base,2)}" )
print(f"Atlanta --> Change in Response variable from the base line: {atl_growth}%")

dal_val_y = dal_data[[target_column]].values
dal_val_x = dal_data.drop(target_column, axis=1).values
# Scale the data
scaled_val_x = scaler_x.fit_transform(dal_val_x)
scaled_val_y = scaler_y.fit_transform(dal_val_y)
last_x_vals = dal_val_x[-1:]
next_x_vals = last_x_vals.copy()
# print("DAL nextx: ", next_x_vals)
print(config.keys())
updated_next_x_vals = update_next_x_vals(next_x_vals, config, config.keys())

dal_y_val, dal_growth = predict_next_value(dal_model_hy_cnn, updated_next_x_vals, dal_base)
# Display the predicted response variable
print(f"Dallas --> Response variable value for the given changes in the Predictor values:{np.round(dal_y_val,2)} with base value:{np.round(dal_base,2)}" )
print(f"Dallas --> Change in Response variable from the base line: {dal_growth}%")


lax_val_y = lax_data[[target_column]].values
lax_val_x = lax_data.drop(target_column, axis=1).values
# Scale the data
scaled_val_x = scaler_x.fit_transform(lax_val_x)
scaled_val_y = scaler_y.fit_transform(lax_val_y)
last_x_vals = lax_val_x[-1:]
next_x_vals = last_x_vals.copy()
# print("LAX nextx: ", next_x_vals)

updated_next_x_vals = update_next_x_vals(next_x_vals, config, config.keys())

lax_y_val, lax_growth = predict_next_value(lax_model_hy_cnn, updated_next_x_vals, lax_base)
# Display the predicted response variable
print(f"Los Angeles --> Response variable value for the given changes in the Predictor values:{np.round(lax_y_val,2)} with base value:{np.round(lax_base,2)}" )
print(f"Los Angeles --> Change in Response variable from the base line: {lax_growth}%")
