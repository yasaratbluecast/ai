from flask import Flask, request, jsonify
# from predict_cost_change_v2 import predict_next_value, update_next_x_vals, read_msa_data, read_config_to_dict
# from tensorflow.keras.models import load_model

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Hello, World! Welcome to my simple Flask app."


@app.route('/models', methods=['POST'])
def models():
    return jsonify({
        "models": [
            "dallas",
            "atlanta",
            "los angeles"
        ]
    })

# @app.route('/submit', methods=['POST'])
# def submit():
#     name = request.get_json()  # Get the 'name' field from the form
#     target_column = 'cpi_food_away_from_home'
#     print(name)
#     if not name or 'model' in name:
#         print("Model is there")
#         if name['model'] == "dallas":
#             config = name['data']
#             dal_base = 252.8825
#             dal_model_hy_cnn = load_model('model/dal_model_hy_cnn_v2.keras')
#             dal_piz_file = "data/Dallas-pizzeria_micro_econ.csv"
#             dal_data = read_msa_data(dal_piz_file)
#             dal_val_x = dal_data.drop(target_column, axis=1).values
#             last_x_vals = dal_val_x[-1:]
#             next_x_vals = last_x_vals.copy()
#             updated_next_x_vals = update_next_x_vals(next_x_vals, config, config.keys())
#             dal_y_val, dal_growth = predict_next_value(dal_model_hy_cnn, updated_next_x_vals, dal_base)
#             dal_y_val = float(dal_y_val)
#             dal_growth = float(dal_growth)
#             return jsonify({"cpi": dal_y_val, "growth": dal_growth}), 200
#         elif name['model'] == "atlanta":
#             config = name['data']
#             atl_base = 245.09427
#             atl_model_hy_cnn = load_model('model/atl_model_hy_cnn_v2.keras')
#             atl_piz_file = "data/Atlanta-pizzeria_micro_econ.csv"
#             atl_data = read_msa_data(atl_piz_file)
#             atl_val_x = atl_data.drop(target_column, axis=1).values
#             last_x_vals = atl_val_x[-1:]
#             next_x_vals = last_x_vals.copy()
#             updated_next_x_vals = update_next_x_vals(next_x_vals, config, config.keys())
#             atl_y_val, atl_growth = predict_next_value(atl_model_hy_cnn, updated_next_x_vals, atl_base)
#             atl_y_val = float(atl_y_val)
#             atl_growth = float(atl_growth)
#             return jsonify({"cpi": atl_y_val, "growth": atl_growth}), 200
#         elif name['model'] == "los angeles":
#             config = name['data']
#             lax_base = 268.20584
#             lax_model_hy_cnn = load_model('model/lax_model_hy_cnn_v2.keras')
#             lax_piz_file = "data/LosAngeles-pizzeria_micro_econ.csv"
#             lax_data = read_msa_data(lax_piz_file)
#             lax_val_x = lax_data.drop(target_column, axis=1).values
#             last_x_vals = lax_val_x[-1:]
#             next_x_vals = last_x_vals.copy()
#             updated_next_x_vals = update_next_x_vals(next_x_vals, config, config.keys())
#             lax_y_val, lax_growth = predict_next_value(lax_model_hy_cnn, updated_next_x_vals, lax_base)
#             lax_y_val = float(lax_y_val)
#             lax_growth = float(lax_growth)
#             return jsonify({"cpi": lax_y_val, "growth": lax_growth}), 200
#         else:
#             return jsonify({"Message": "Please specify a valid model name."}), 400
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
