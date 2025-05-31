# Import libraries
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.stattools import pacf
from collections import defaultdict, deque
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from datetime import datetime

import warnings
warnings.simplefilter("ignore", UserWarning)

################################################################################## 
# Function for input data 
################################################################################## 

def load_excel_if_needed(data, relative_path):
    """
    Loads Excel file from a given path if data is None, empty, or missing required columns.

    Parameters:
        data (pd.DataFrame or None): Existing dataframe to validate or replace.
        relative_path (list of str): Relative path to the Excel file.

    Returns:
        pd.DataFrame: A valid dataframe loaded from file or the input data if already valid.
    """

    base_path = os.path.dirname(__file__) 
    path = os.path.join(base_path, *relative_path)
    required_columns = {'Bilagstype', 'Afsendelsesdato', 'Antal', 'Varestatistikgruppe 2'}

    if data is None or data.empty:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_excel(path)
    
    if required_columns.issubset(data.columns):
        return data
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_excel(path)
    

def take_inputs(sales_data: pd.DataFrame = None, 
                component_data:pd.DataFrame = None,
                inventory_data:pd.DataFrame = None,
                machine_dict: dict[str, list[int | str]] = None,
                machine_types: dict = None,
                aggregation_method:str = None,
                forecast_periods: int = None,
                n_lags:int = None,
                varegrupper:dict = None,
                non_representative_years: list[int] = None):
    """
    Loads and validates user inputs for the forecasting pipeline.
    
    Parameters:
        sales_data: Raw sales data. Loaded from file if None or invalid.
        component_data: Loaded from file if None or invalid.
        inventory_data: Loaded from file if None or invalid.
        machine_dict: Mapping of machine types to product numbers or prefixes.
        machine_types: Mapping of machine types to their components.
        aggregation_method: Time aggregation method ('month' or 'week'). Defaults to 'month'.
        forecast_periods: Number of periods to forecast. Defaults depend on aggregation.
        n_lags: Number of lags for feature engineering. Defaults depend on aggregation.
        varegrupper: Component categorization mapping.
        non_representative_years: Years to exclude from model training. Defaults to [2020].

    Returns:
        Tuple of validated inputs: (sales_data, component_data, inventory_data, machine_dict, aggregation_method, n_lags, varegrupper)
    """

    sales_data = load_excel_if_needed(sales_data, ["DataDefault", "DataRaw", "Data_til_forecast.xlsx"])
    component_data = load_excel_if_needed(component_data, ["DataDefault", "DataRaw", "Stykkeliste.xlsx"])
    inventory_data = load_excel_if_needed(inventory_data, ["DataDefault", "DataRaw", "Vareinfo liste.xlsx"])

    if not machine_dict:
        machine_dict = {
            'opvasker': [8100, 8105, 9110, 9115, 9117, 9120, 9130, 8150, 8160],
            'kassevasker': [200, 201, 300, 220],
            'pladerenser_type9020': [9020],
            'pladerenser_type3': ['60', '70', '90', '95']}  # Prefix

    if not machine_types:
        machine_types = {
        "kassevasker": ["Slanger", "Doseringspumpe", "Vaskepumpe", "Styring", "Varmelegeme"],
        "opvasker": ["Slanger", "Doseringspumpe", "Vaskepumpe", "Skyllepumpe", "Sæbepumpe", "Drænpumpe", "Styring"],
        "pladerenser_type9020": ["Doseringspumpe", "Styring", "Børster", "Valser", "Olieakser"],
        "pladerenser_type3":["Styring", "Børster", "Valser", "Olieakser"]}
        
    if not aggregation_method:
        aggregation_method = "month"

    if not n_lags:
        if aggregation_method == "month":
            n_lags = 7
        elif aggregation_method == "week":
            n_lags = 27
        else:
            aggregation_method = "month"
            n_lags = 7

    if not varegrupper:
        varegrupper = { # component: {varegruppekode: [produktgruppekode]}
                    "Doseringspumpe": {"pumper": ["dosering"]},
                    "Vaskepumpe": {"pumper": ["vaske"]},
                    "Skyllepumpe": {"pumper": ["skylle"]},
                    "Sæbepumpe": {"pumper": ["sæbe"]},
                    "Drænpumpe": {"pumper": ["dræn"]},
                    "Vakuumpumpe": {"pumper": ["vakum"]},
                    "Valser": {"valser": ["stål", "gummi", "658", "over", "rf"]},
                    "Olieakser": {"aksel": ["olie"]},
                    "Børster": {"reservedel": ["børste"]},
                    "Varmelegeme": {"reservedel": ["varmelegem"]},
                    "Styring": {"tc":["styring"]},
                    "Slanger":{"slange":[]}}
    
    if not non_representative_years:
        non_representative_years = [2020]

    if not forecast_periods:
        if aggregation_method == "month":
            forecast_periods = 3
        elif aggregation_method == "week":
            forecast_periods = 12
        else:
            aggregation_method = "month"
            forecast_periods = 3
    return sales_data, component_data, inventory_data, machine_dict, machine_types, aggregation_method, n_lags, varegrupper, non_representative_years, forecast_periods


################################################################################## 
# Functions for preparing the data 
################################################################################## 

def find_machine_type(machine_dict: dict[str, list[int | str]] = None, product_number: int | str = None) -> str:
    """
    Identify the machine type based on a product number.

    Parameters:
        machine_dict: Optional dict mapping machine types to lists of product numbers or string prefixes.
        product_number: The product number to classify.

    Returns:
        The machine type as a string, or "unknown model" if no match is found.

    Example mapping:
        {
            'opvasker': [8100, 8105, 9110],
            'pladerenser_type3': ['60', '70']  # Matches by prefix
        }
    """

    if product_number is None:
        return "unknown model"

    digits_only = "".join([item for item in str(product_number) if item.isdigit()])

    if not digits_only:
        return "unknown model"
    else:
        product_number = int(digits_only)

        # Check if any integer values matches the product number
        for key, values in machine_dict.items():
            if any(isinstance(v, int) for v in values) and product_number in values:
                return key

        # Check if any string values matches the product number
        product_str = str(product_number)
        for key, values in machine_dict.items():
            if any(isinstance(v, str) for v in values):
                if any(product_str.startswith(prefix) for prefix in values):
                    return key

    return "unknown model"


def remove_non_representative_years(data:pd.DataFrame, years:list[int]):
    """
    Removes rows from a DataFrame based on a list of years.

    Parameters:
        data: DataFrame with a DatetimeIndex
        years: List of years (int) to exclude from the dataset

    Returns:
        Filtered DataFrame without the specified years
    """
    if years:
        if all(isinstance(year, int) for year in years):
            data = data[~data.index.year.isin(years)]
        else:
            None
    else:
        None
    return data


def prepare_sales_data(data:pd.DataFrame, machine_dict:dict = None, aggregation_method:str = 'month', non_representative_years: list[int] = None):
    """
    Prepares sales data for analysis by filtering, cleaning, categorizing, and aggregating.

    Parameters:
        data: Raw sales data containing the columns: 'Bilagstype' (document type), 'Afsendelsesdato' (shipping date), 'Antal' (quantity), and 'Varestatistikgruppe 2' (product number).
        machine_dict: Mapping of machine types to product numbers or prefixes. 
        aggregation_method: Time aggregation method, either 'month' or 'week'. Defaults to 'month'.
        non_representative_years: List of years to exclude from the aggregated data, e.g. years with unusual sales patterns.

    Returns:
        pd.DataFrame: Aggregated demand indexed by date with machine-type columns.
    """
    
    # Replace NaT and NaN values with 'None'
    data.replace({pd.NaT: 'None'}, inplace=True)
    data.replace({'NaN': 'None'}, inplace=True)

    data = data[['Bilagstype', 'Afsendelsesdato', 'Antal', 'Varestatistikgruppe 2']]

    data = data.loc[(data['Bilagstype'] == 'Salgsfaktura')]
    data['Afsendelsesdato'] = pd.to_datetime(data['Afsendelsesdato'], format='%Y-%m-%d')
    data = data[['Afsendelsesdato', 'Antal', 'Varestatistikgruppe 2']].copy()

    data = data.set_index('Afsendelsesdato', drop = False).sort_index()
    data['Machine'] = data['Varestatistikgruppe 2'].apply(lambda x: find_machine_type(machine_dict=machine_dict, product_number=x))

    # Group the data such that each column represents the demand for each machine
    if str(aggregation_method).lower() == 'month':
        freq = 'ME'
    elif str(aggregation_method).lower() == 'week':
        freq = 'W'
    else:
        freq = 'ME'
        print('Neither "month" or "week" has been chosen as the aggregation_method. The default value, month, is chosen')

    demand_grouped = (data.groupby(
        [pd.Grouper(key='Afsendelsesdato', freq=freq), 'Machine'])['Antal']
        .sum()
        .unstack(fill_value=0))

    if non_representative_years:
        demand_grouped = remove_non_representative_years(data=demand_grouped, years=non_representative_years)
    else:
        None
    
    return demand_grouped


def create_dataset_with_lagged_features(data: pd.DataFrame, 
                           n_lags: int = None, 
                           aggregation_method: str = None):
    """
    Computes significant PACF lags and returns a DataFrame with lagged features.

    Parameters:
        data: pd.DataFrame with a single column and a DatetimeIndex.
        n_lags: Number of lags to include. Defaults by aggregation_method.
        aggregation_method: 'month' or 'week' to set lag limits.
    
    Returns:
        tuple: (DataFrame with lagged features, list of selected lag indices)
    """

    # Validate input
    if data.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one column.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex.")

    # Default aggregation method
    method = (aggregation_method or "month").lower()

    # Maximum allowed lags per method
    max_lags_by_method = {
        'month': 12,
        'week': 52
    }
    
    if method not in max_lags_by_method:
        method = 'month'
    
    if n_lags is None:
        n_lags = max_lags_by_method[method]

    # Limit n_lags in case it is larger than the maximum allowed number
    max_allowed = max_lags_by_method[method]
    if n_lags > max_allowed:
        n_lags = max_allowed

    # Find the pacf values and select the most important ones
    max_pacf_lags = max_lags_by_method[method]

    pacf_values = pacf(data.values.flatten(), nlags=max_pacf_lags)
    selected_lags = list(np.sort(np.argsort(np.abs(pacf_values))[-n_lags:][::-1])[1:])

    # Create new dataframe
    lagged_df = pd.DataFrame(index=data.index)
    lagged_df['Actual_value'] = data.iloc[:, 0]

    for lag in selected_lags:
        lagged_df[f"lag_{lag}_{method}"] = data.iloc[:, 0].shift(lag)

    # Drop rows with NaN (from shift)
    lagged_df = lagged_df.iloc[max_pacf_lags:]
    lagged_df = lagged_df.dropna()

    return lagged_df, selected_lags


################################################################################## 
# Functions for training an ML forecasting model 
################################################################################## 

def create_datasets_cv(cv_type: str, data: pd.DataFrame, aggregation_method: str = None):
    """
    Splits data into training and rolling test sets based on the chosen cross-validation type.

    Parameters:
        cv_type: Cross-validation strategy, either "block" or "gap_split".
        data: DataFrame indexed by date, containing 'Actual_value' column.
        aggregation_method: Time aggregation method, "month" or "week" (default: "month").

    Returns:
        Dictionary of train/test splits for each CV fold and rolling window.
    """

    if not aggregation_method:
        aggregation_method = "month"
    method = aggregation_method.lower()

    datasets = {}
    try:
        cv = cv_type.lower()
        data = data.dropna()

        all_years = sorted(data.index.year.unique())

        train_splits = []
        test_splits = []

        if cv == "block":
            n_splits = len(all_years) - 1
            for i in range(0, n_splits, 2):  # step by 2 to ensure that train and test never overlap
                train_year = all_years[i]
                test_year = all_years[i + 1]
                train_splits.append([train_year])
                test_splits.append([test_year])
        elif cv == "gap_split":
            n_splits = len(all_years) - 2  # leave room for gap
            for i in range(n_splits):
                train_splits.append([all_years[i]])
                test_splits.append([all_years[i + 2]])
        else:
            raise ValueError("cv_type not recognized")

        for split_id, (train_years, test_years) in enumerate(zip(train_splits, test_splits), 1):
            test_data = data[data.index.year.isin(test_years)]

            if method == "month":
                test_step = pd.DateOffset(months=3)
                train_offset = pd.DateOffset(months=12)
                gap_duration = pd.DateOffset(months=3)
                first_test_date = test_data.index.min().replace(day=1)
            elif method == "week":
                test_step = pd.DateOffset(weeks=12)
                train_offset = pd.DateOffset(weeks=52)
                gap_duration = pd.DateOffset(weeks=12)
                first_test_date = test_data.index.min() - pd.to_timedelta(test_data.index.min().weekday(), unit='D')
            else:
                raise ValueError("Unsupported aggregation method. Use 'month' or 'week'.")

            last_test_date = test_data.index.max()
            roll_id = 1

            while first_test_date + test_step - pd.DateOffset(days=1) <= last_test_date:
                test_start = first_test_date
                test_end = test_start + test_step - pd.DateOffset(days=1)

                if cv == "gap_split":
                    train_end = test_start - gap_duration - pd.DateOffset(days=1)
                else:
                    train_end = test_start - pd.DateOffset(days=1)

                train_start = train_end - train_offset + pd.DateOffset(days=1)

                train = data[(data.index >= train_start) & (data.index <= train_end)]
                test = data[(data.index >= test_start) & (data.index <= test_end)]

                if len(train) == 0 or len(test) == 0:
                    break

                X_train = train.drop(columns=['Actual_value', 'year', method], errors='ignore')
                y_train = train[['Actual_value']]
                X_test = test.drop(columns=['Actual_value', 'year', method], errors='ignore')
                y_test = test[['Actual_value']]

                datasets[f"Split_{split_id}_Roll_{roll_id}"] = {
                    "train_period": (train_start, train_end),
                    "test_period": (test_start, test_end),
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }

                first_test_date += test_step
                roll_id += 1

        return datasets

    except Exception as e:
        print(f"Error: {e}")
        return None
    

def train_forecasting_model(data: pd.DataFrame, aggregation_method: str = None):
    """
    Performs hyperparameter tuning and cross-validation with different CV types to find the "best"/most robust parameter combination.
    Then trains the a Gradient Boosting model on the whole dataset using the "best" parameter combination.

    Parameters:
        data: DataFrame with lagged features and 'Actual_value' column.
        aggregation_method: Time aggregation method, 'month' or 'week'.

    Returns:
        model_dict: dictionary containing final model, parameters, mae, mase and training data.
    """

    if aggregation_method.lower() == "month":
        sp = 6
    else:
        sp = 1

    # Define cross-validation types and hyperparameter grid
    cv_types = ["block", "gap_split"]
    loss_options = ['absolute_error']
    learning_rates = [0.005, 0.01, 0.1]
    min_samples_leafs = [1, 2]
    max_depths = [2,3]
    n_estimators_list = [75, 100, 125]

    best_result = None
    results = {}
    model_number = 0

    for l in loss_options:
        for lr in learning_rates:
            for leaf in min_samples_leafs:
                for depth in max_depths:
                    for n_est in n_estimators_list:
                        parameters = [l, lr, leaf, depth, n_est]

                        for cv_type in cv_types:
                            datasets = create_datasets_cv(cv_type=cv_type, data=data, aggregation_method=aggregation_method)
                            if datasets is None or not datasets:
                                print(f"Skipping combination {parameters} with CV type '{cv_type}'. No valid dataset.")
                                continue

                            maes = []
                            mases = []
                            all_y_pred = []
                            all_y_true = []
                            df_all_results = []

                            for split in datasets.values():
                                X_train, y_train = split['X_train'], split['y_train']
                                X_test, y_test = split['X_test'], split['y_test']

                                # Scale data
                                scaler_X = MinMaxScaler()
                                X_train_scaled = scaler_X.fit_transform(X_train)
                                X_test_scaled = scaler_X.transform(X_test)

                                scaler_y = MinMaxScaler()
                                y_train_scaled = scaler_y.fit_transform(y_train)
                                y_test_scaled = scaler_y.transform(y_test)

                                # Train model
                                model = GradientBoostingRegressor(
                                    loss=l,
                                    learning_rate=lr,
                                    min_samples_leaf=leaf,
                                    max_depth=depth,
                                    n_estimators=n_est,
                                    random_state=42
                                )
                                model.fit(X_train_scaled, y_train_scaled)

                                # Predict and evaluate
                                y_pred = model.predict(X_test_scaled)
                                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                                y_true = scaler_y.inverse_transform(y_test_scaled)

                                # Skip if arrays are empty
                                if y_train.empty or y_test.empty or len(y_pred) == 0 or len(y_true) == 0:
                                    continue
                                
                                mae = mean_absolute_error(y_true, y_pred)
                                maes.append(mae)

                                if len(y_train) < sp:
                                    mase = np.nan
                                else:
                                    mase = mean_absolute_scaled_error(y_true=y_true, y_pred=y_pred, y_train=y_train, sp=sp)

                                mases.append(mase)

                                all_y_pred.extend(y_pred.flatten())
                                all_y_true.extend(y_true.flatten())
                                df_results = pd.DataFrame({'y_true': y_true.ravel(),
                                                           'y_pred': y_pred.ravel()},
                                                           index=y_test.index)

                                df_all_results.append(df_results)

                            avg_mae = np.mean(maes)
                            min_mae = np.min(maes)

                            avr_mase = np.mean(mases)
                            min_mase = np.min(mases)
                            model_number += 1

                            mean_y_pred = np.mean(all_y_pred)
                            std_y_pred = np.std(all_y_pred)

                            mean_y_true = np.mean(all_y_true)
                            std_y_true = np.std(all_y_true)

                            df_all_predictions = pd.concat(df_all_results).sort_index()

                            results[str(model_number)] = {
                                'cv_type': cv_type,
                                'parameters': parameters,
                                'mae': avg_mae,
                                'min_mae': min_mae,
                                'mase': avr_mase,
                                'min_mase':min_mase,
                                'mean_y_pred': mean_y_pred,
                                'std_y_pred': std_y_pred,
                                'mean_y_true': mean_y_true,
                                'std_y_true': std_y_true,
                                'df_all_predictions': df_all_predictions
                            }

                            # Update best result
                            if best_result is None or (avg_mae < best_result['mae'] and avr_mase < best_result['mase']) :
                                best_result = results[str(model_number)]

    # Train final model
    data_train = data.copy()
    X = data_train.drop(columns=['Actual_value'])
    y = data_train[['Actual_value']]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    best_params = best_result['parameters']
    best_mae = best_result['mae']
    best_mase = best_result['mase']
    mean_y_pred = best_result['mean_y_pred']
    mean_y_true = best_result['mean_y_true']
    std_y_pred = best_result['std_y_pred']
    std_y_true = best_result['std_y_true']
    df_all_predictions = best_result['df_all_predictions']
    
    final_model = GradientBoostingRegressor(
        loss=best_params[0],
        learning_rate=best_params[1],
        min_samples_leaf=best_params[2],
        max_depth=best_params[3],
        n_estimators=best_params[4],
        random_state=42
    )

    final_model.fit(X_scaled, y_scaled.ravel())

    # Package everything
    model_dict = {
        'model': final_model,
        'parameters': best_result['parameters'],
        'cv_type': best_result['cv_type'],
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'X_train': X,
        'y_train': y,
        'mae': best_mae,
        'mase': best_mase,
        'std_pred': std_y_pred,
        'std_true': std_y_true,
        'mean_y_pred': mean_y_pred,
        'mean_y_true': mean_y_true,
        'df_results': df_all_predictions
    }

    print("Best model trained using", best_result['cv_type'], "Parameters", best_result['parameters'], "MAE:", round(best_mae, 4), "MASE:", round(best_mase, 4))
    return model_dict


def plot_predictions(model_dict, filename='full_year_forecast.png', key=None, input=False, timestamp=None):
    """
    Plots actual vs. predicted values from CV test folds and saves the plots to file.
    Creates separate plots for each continuous year range if there are gaps in the timeline.

    Parameters:
        model_dict: Dictionary containing prediction results and metrics.
        filename: Name of the file to save the plot as.
        key: Optional string to include in the plot title.
        input: Bool, if True saves to input folder, else default folder.
        timestamp: Optional string to append to filename for uniqueness.

    Returns:
        None. Saves plot files in the 'PredictionPlots' directory.
    """
    df = model_dict['df_results'].copy()
    df = df.sort_index()

    if timestamp is not None:
        timestamp = timestamp.replace(":", "-").replace("T", "_")
    else:
        timestamp = ''

    years = df.index.year
    year_diffs = np.diff(sorted(set(years)))
    split_years = [i for i, diff in enumerate(year_diffs) if diff > 1]

    # Determine year groups
    year_ranges = []
    all_years = sorted(set(years))
    start_idx = 0
    for split_idx in split_years:
        end_idx = split_idx + 1
        year_ranges.append(all_years[start_idx:end_idx])
        start_idx = end_idx
    year_ranges.append(all_years[start_idx:])

    base_path = os.path.dirname(__file__)
    folder_name = "DataInputs" if input else "DataDefault"
    plot_dir = os.path.join(base_path, folder_name, "PredictionPlots")
    os.makedirs(plot_dir, exist_ok=True)

    for idx, year_group in enumerate(year_ranges):
        df_subset = df[df.index.year.isin(year_group)]

        plt.figure(figsize=(12, 6))
        plt.plot(df_subset.index, df_subset['y_true'], label='Actual', marker='o')
        plt.plot(df_subset.index, df_subset['y_pred'], label='Predicted', marker='x')
        title_base = f'{key}: ' if key else ''
        plt.title(f'{title_base}Predicted vs Actual ({year_group[0]}–{year_group[-1]})')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Metrics annotation
        mae = model_dict.get('mae')
        mase = model_dict.get('mase')
        std_pred = model_dict.get('std_pred')
        std_true = model_dict.get('std_true')
        mean_pred = model_dict.get('mean_y_pred')
        mean_true = model_dict.get('mean_y_true')
        parameters = model_dict.get('parameters')

        if all(v is not None for v in [mae, mase, std_pred, std_true, mean_pred, mean_true, parameters]):
            plt.text(
                0.01, 0.99,
                f"MAE: {round(mae,1):.2f}\nMASE: {round(mase,1):.2f}\n"
                f"Std (pred): {round(std_pred,1)}\nStd (true): {round(std_true,1)}\n"
                f"Mean (pred): {round(mean_pred,1)}\nMean (true): {round(mean_true,1)}\n"
                f"Parameters: {parameters}",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="gray", alpha=0.5)
            )

        # Save each figure with year range and optional timestamp
        name, ext = os.path.splitext(filename)
        year_suffix = f"{year_group[0]}_{year_group[-1]}"
        ts_suffix = f"_{timestamp}" if timestamp else ""
        full_filename = f"{name}_{year_suffix}{ts_suffix}.png"
        full_path = os.path.join(plot_dir, full_filename)

        plt.savefig(full_path)
        plt.close()

################################################################################## 
# Functions for forecasting 
##################################################################################

def prepare_machine_data_for_prediction(data: pd.DataFrame, aggregation_method: str = 'month'):
    """
    Prepares the most recent 12 months or 52 weeks of data for prediction by adding time-based features.

    Parameters:
        data: DataFrame with a DatetimeIndex.
        aggregation_method: Time aggregation method, 'month' or 'week'.

    Returns:
        DataFrame sorted in reverse chronological order with added features.
    """

    data_predict = data.copy()
    data_predict['year'] = data_predict.index.year

    if aggregation_method == 'month':
        cutoff_date = data_predict.index.max() - pd.DateOffset(months=12)
        data_predict = data_predict[data_predict.index > cutoff_date]
        data_predict['month'] = data_predict.index.month
        length = data_predict.shape[0]
        data_predict['lag'] = list(range(1, length+1))

    elif aggregation_method == 'week':
        cutoff_date = data_predict.index.max() - pd.DateOffset(weeks=52)
        data_predict = data_predict[data_predict.index > cutoff_date]
        data_predict['week'] = data_predict.index.isocalendar().week
        length = data_predict.shape[0]
        data_predict['lag'] = list(range(1, length+1))
    else:
        raise ValueError("aggregation_method must be either 'month' or 'week'")

    data_predict = data_predict.sort_index(ascending=False)

    return data_predict


def prepare_component_data_for_prediction(data:pd.DataFrame, machine_dict:dict = None):
    """
    Maps component product numbers to machine types and filters non-null components.

    Parameters:
        data: DataFrame with a 'Styklistevarenr.' column.
        machine_dict: Dictionary mapping product numbers to machine types.

    Returns:
        Filtered DataFrame with an added 'Machine' column.
    """

    component_data = data[data['Styklistevarenr.'].isna() == False]
    component_data['Machine'] = component_data['Styklistevarenr.'].apply(lambda x: find_machine_type(machine_dict=machine_dict, product_number=x))
    return component_data


def predict_machine_demand(model_dict: dict, 
                           data_for_prediction: pd.DataFrame, 
                           pacf_lags: list[int], 
                           forecast_unit: str = 'month', 
                           forecast_periods: int = None):
    """
    Predicts future machine demand using a trained model and recursive forecasting.

    Parameters:
        model_dict: Dictionary with a trained model and corresponding scalers to use for normalization.
        data_for_prediction: DataFrame containing latest historical observations and features.
        pacf_lags: List of significant lags selected from PACF analysis.
        forecast_unit: Time unit for forecasting ('month' or 'week').
        forecast_periods: Number of future periods to forecast. 
                          Defaults to 3 months or 12 weeks based on forecast_unit.

    Returns:
        DataFrame with forecasted demand values and predicted timestamps as index.
    """

    if forecast_periods is None:
        forecast_periods = 12 if forecast_unit == 'week' else 3

    model = model_dict['model']
    data = data_for_prediction.copy()
    scaler_X = model_dict['scaler_X']
    scaler_y = model_dict['scaler_y']

    method = forecast_unit.lower()
    column = data.columns[0]
    machine_demand_df = pd.DataFrame()

    for i in range(forecast_periods):
        # Get latest date
        latest_date = data.index.max()

        # Calculate new forecast date
        if forecast_unit == 'month':
            new_date = latest_date + pd.DateOffset(months=1)
        elif forecast_unit == 'week':
            new_date = latest_date + pd.DateOffset(weeks=1)
        else:
            raise ValueError("forecast_unit must be either 'month' or 'week'")

        # Build feature vector from selected PACF lags
        feature_dict = {}
        for lag in pacf_lags:
            lag_date = latest_date - pd.DateOffset(**{f"{method}s": lag})
            try:
                lag_value = data.loc[lag_date, column]
            except KeyError:
                lag_value = 0  # Default if missing
            feature_dict[f"lag_{lag}_{method}"] = lag_value

        # Add time-based features
        feature_dict['year'] = new_date.year
        if method == 'month':
            feature_dict['month'] = new_date.month
        elif method == 'week':
            feature_dict['week'] = new_date.isocalendar().week

        # Format into DataFrame
        X_pred = pd.DataFrame(feature_dict, index=[new_date])
        X_pred = X_pred.reindex(columns=scaler_X.feature_names_in_, fill_value=0)
        X_scaled = scaler_X.transform(X_pred)

        # Predict and inverse scale
        forecast_i = model.predict(X_scaled).reshape(-1, 1)
        forecast_i = scaler_y.inverse_transform(forecast_i)

        # Store forecast
        forecast_df = pd.DataFrame({
            "forecasted demand": forecast_i.flatten()
        }, index=[new_date])
        machine_demand_df = pd.concat([machine_demand_df, forecast_df])

        # Append prediction to data for recursive lags
        new_row = pd.DataFrame({
            column: forecast_i.flatten(),
            'year': new_date.year,
            method: new_date.month if method == 'month' else new_date.isocalendar().week,
            'lag': 1
        }, index=[new_date])
        data = pd.concat([data, new_row])
        data.loc[data.index != new_date, 'lag'] += 1

    machine_demand_df["forecasted demand"] = machine_demand_df["forecasted demand"].astype(int)
    return machine_demand_df


def convert_machine_to_component_demand(machine_demand_df: pd.DataFrame, machine_types: dict):    
    """
    Converts forecasted machine-level demand to component-level demand using a machine-component mapping.

    Parameters:
        machine_demand_df: DataFrame with forecasted demand per machine type.
                           Index should represent time (e.g. datetime), and columns should include machine types.
        machine_types: Dictionary where keys are machine types and values are lists of components required per machine.

    Returns:
        DataFrame with forecasted component demand, indexed by time.
        Time-related columns ('Month', 'Week') from input are preserved if present.
    """
    
    # Compute component demand by summing machine demands
    component_demand = defaultdict(lambda: np.zeros(machine_demand_df.shape[0], dtype=float))
    for machine, components in machine_types.items():
        if machine not in machine_demand_df.columns:
            continue  # skip unknown machines
        for component in components:
            component_demand[component] += machine_demand_df[machine].values

    # Convert to DataFrame
    component_demand_df = pd.DataFrame(component_demand, index=machine_demand_df.index)

    # Preserve time-related columns if present
    if "Month" in machine_demand_df.columns:
        component_demand_df["Month"] = machine_demand_df["Month"]
    if "Week" in machine_demand_df.columns:
        component_demand_df["Week"] = machine_demand_df["Week"]

    return component_demand_df


################################################################################## 
# Functions for inventory optimization 
################################################################################## 

def convert_to_days(value, time_units={'d': 1, 'w': 7, 'm': 4 * 7}):
    """
    Converts a string-based time duration into days.

    Parameters:
        value: A string representing time with a unit suffix.
               Supported units are:
               - 'd' for days
               - 'w' for weeks
               - 'm' for months (approximated as 4 weeks = 28 days)
        time_units: Optional dictionary mapping unit characters to their equivalent in days.
                    Default is {'d': 1, 'w': 7, 'm': 28}.

    Returns:
        The number of days as an integer, or NaN if the input is invalid or improperly formatted.
    """

    if isinstance(value, str):
        unit = value[-1]
        if unit in time_units:
            try:
                return int(value[:-1]) * time_units[unit]
            except ValueError:
                return np.nan
    return np.nan


def compute_lead_time(data: pd.DataFrame, 
                      varegrupper: dict,
                      time_unit_mapping: dict = None,
                      aggregation_method:str ="month"):
    """
    Computes lead time values per defined varegruppe using a hierarchical fallback strategy.

    Parameters:
        data: A DataFrame containing product data, including lead times and product codes.
        varegrupper: A dictionary mapping group keys to pairs of varekategorikoder and produktgruppekoder.
        time_unit_mapping: Optional dictionary mapping time unit characters (e.g., 'd', 'w', 'm') to days.
                           Defaults to {'d': 1, 'w': 7, 'm': 28}.
        aggregation_method: Time aggregation method, 'month' or 'week'.

    Returns:
        A dictionary where each key corresponds to a varegruppe and each value is a list of computed lead times,
        converted to either months or weeks (depending on aggregation_method), with a minimum of 1 unit per entry.
    """
    
    time_unit_mapping = time_unit_mapping or {'d': 1, 'w': 7, 'm': 28}
    
    lead_time_values = {}

    data = data[['Varenummer', 'Beskrivelse', 'Beskrivelse 2', 'Produktbogføringsgruppe', 'Varekategorikode', 'Produktgruppekode', 'Leveringstid Varekort']]

    for column in data.columns:
        data[column] = data[column].str.lower()
    
    for key in varegrupper.keys():
        for varekategorikode, produktgruppekode in varegrupper[key].items():
            lead_time_values_raw1 = data[(data['Varekategorikode'] == varekategorikode) & 
                                        (data["Produktgruppekode"].isin(produktgruppekode))
                                        ]['Leveringstid Varekort'].dropna().values
            
            if len(lead_time_values_raw1) > 0:
                lead_time_values_raw = lead_time_values_raw1
            else:
                lead_time_values_raw2 = data[(data['Varekategorikode'] == varekategorikode) & 
                                            (data["Beskrivelse"].str.contains('|'.join(produktgruppekode), case=False, na=False))
                                            ]['Leveringstid Varekort'].dropna().values
                
                if len(lead_time_values_raw2) > 0:
                    lead_time_values_raw = lead_time_values_raw2
                else:
                    lead_time_values_raw3 = data[data['Varekategorikode'] == varekategorikode
                                               ]['Leveringstid Varekort'].values
                    
                    if len(lead_time_values_raw3) > 0:
                        lead_time_values_raw = lead_time_values_raw3
                    else:
                        lead_time_values_raw = []

            if aggregation_method == "month":
                lead_time_values[key] = [max([1, convert_to_days(value, time_unit_mapping)//28]) for value in lead_time_values_raw if pd.notna(value)]
            elif aggregation_method == "week":
                lead_time_values[key] = [max([1,(convert_to_days(value, time_unit_mapping)//4)]) for value in lead_time_values_raw if pd.notna(value)]
            else:
                lead_time_values[key] = []

    return lead_time_values


def calculate_lead_times(lead_times:dict, aggregation_method='month'):
    """
    Calculates average and maximum lead times for each group in the input.

    Parameters:
        lead_times: Dictionary with group keys mapped to lists of lead time values.
        aggregation_method: Defines the time granularity, either 'month' or 'week'.

    Returns:
        Two dictionaries:
        - One with the average lead time per group.
        - One with the maximum lead time per group.
        
        If no valid values are found for a group:
        - Default is 1 (avg) and 2 (max) for 'month'.
        - Default is 4 (avg) and 8 (max) for 'week'.
    """
    
    if aggregation_method.lower() == 'month':
        lead_times_avr = {
            key: int(np.nanmean([v for v in values if not pd.isna(v)])) if any(pd.notna(v) for v in values)
                 else 1
            for key, values in lead_times.items()
        }

        lead_times_max = {
            key: int(np.nanmax([v for v in values if not pd.isna(v)])) if any(pd.notna(v) for v in values)
                 else 2
            for key, values in lead_times.items()
        }
    
    elif aggregation_method.lower() == 'week':
        lead_times_avr = {
            key: int(np.nanmean([v for v in values if not pd.isna(v)])) if any(pd.notna(v) for v in values)
                 else 4
            for key, values in lead_times.items()
        }

        lead_times_max = {
            key: int(np.nanmax([v for v in values if not pd.isna(v)])) if any(pd.notna(v) for v in values)
                 else 8
            for key, values in lead_times.items()
        }
    
    return lead_times_avr, lead_times_max


def safety_stock(max_demand:int, max_lead_time, avr_demand:int, avr_lead_time):
    """
    Calculates the safety stock level based on maximum and average demand and lead time.

    The safety stock is computed as:
        (maximum demand × maximum lead time) − (average demand × average lead time)

    This method accounts for variability in both demand and lead time, helping to buffer against uncertainties.

    Parameters:
        max_demand: Maximum observed or estimated demand per period.
        max_lead_time: Maximum lead time in the same time unit.
        avr_demand: Average demand per period.
        avr_lead_time: Average lead time in the same time unit.

    Returns:
        The recommended safety stock level.
    """
    ss = int((max_demand*max_lead_time)-(avr_demand*avr_lead_time))
    return ss


def simulate_inventory_fixed(component_demand_df, component_types, initial_inventory, lead_times, safety_stock, override_order_qty=None, override_period=None):
    """
    Simulates inventory flow and ordering decisions for multiple components over time.

    Parameters:
        component_demand_df: DataFrame with time-indexed demand per component.
        component_types: Dictionary of components to include in simulation.
        initial_inventory: Dictionary of initial inventory levels per component.
        lead_times: Dictionary with lead times per component.
        safety_stock: Dictionary with safety stock levels per component.
        override_order_qty: Optional, quantity to manually order in the override period.
        override_period: Optional, time period at which to place the override order.

    Returns:
        inventory_history: Dictionary with inventory levels per component over time
        order_history: Dictionary with order quantities per component and time step
        reorder_points: Dictionary with calculated reorder points per component and time step
    """

    common_component_types = [
        component 
        for component in component_types.keys()
        if component in component_demand_df.columns
    ]
    
    inventory = initial_inventory.copy()
    orders_in_transit = {component: deque() for component in common_component_types}  # Track orders (arrival month, qty)
    inventory_history = {component: [] for component in common_component_types}
    order_history = {component: [] for component in common_component_types}
    reorder_points = {component: [] for component in common_component_types}

    for month in range(len(component_demand_df)):
        # Process incoming orders
        for component, order_queue in orders_in_transit.items():
            while order_queue and order_queue[0][0] == month:  # If order arrives this month
                _, qty = order_queue.popleft()
                inventory[component] += qty

        # Register inventory levels before consumption
        for component in common_component_types:
            inventory_history[component].append(inventory[component])

        # Consume inventory based on demand
        for component, demand in component_demand_df.iloc[month].items():
            inventory[component] -= demand

        # Place orders
        for component in common_component_types:
            future_month = int(round(month + lead_times[component]))
            projected_demand = component_demand_df[component].iloc[month+1:future_month+1].sum()
            reorder_point = int(projected_demand + safety_stock[component])
            reorder_points[component].append(reorder_point)

            # Check for manual override
            if override_order_qty is not None and override_period is not None and month == override_period:
                order_qty = override_order_qty
                delivery_month = month + lead_times[component]
                if delivery_month < len(component_demand_df):
                    orders_in_transit[component].append((delivery_month, order_qty))
                    order_history[component].append((month, order_qty))
                else:
                    order_history[component].append((month, 0))
            else:
                if inventory[component] <= reorder_point:
                    order_qty = reorder_point - inventory[component]
                    delivery_month = month + lead_times[component]
                    if delivery_month < len(component_demand_df):
                        orders_in_transit[component].append((delivery_month, order_qty))
                        order_history[component].append((month, order_qty))
                    else:
                        order_history[component].append((month, 0))
                else:
                    order_history[component].append((month, 0))

    return inventory_history, order_history, reorder_points


def update_upload_log(file_name = None, timestamp = None, path = None, aggregation_method = None):
    """
    Updates a JSON log file with information about uploaded files.
    If a file name is provided, it attempts to read the existing log file (or creates a new one if it doesn't exist),
    and appends a new entry containing the file name, timestamp, path, and aggregation method.

    Parameters:
        file_name: Name of the uploaded file.
        timestamp: Timestamp of the upload.
        path: Path to the uploaded file.
        aggregation_method: Aggregation method used ('month' or 'week').

    Returns:
        None
    """
    
    base_path = os.path.dirname(__file__) 
    relative_path = ["Logs", "upload_log.json"]
    log_file = os.path.join(base_path, *relative_path)

    if file_name:
        # Try read the existing file
        try:
            with open(log_file, "r") as f:
                upload_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is empty, start with an empty list
            upload_log = []

        entry = {
            "filename": file_name,
            "timestamp": timestamp,
            "aggregation_method": aggregation_method,
            "path": path
        }

        upload_log.append(entry)

        # Save the updated log in the same JSON-filen
        with open(log_file, "w") as f:
            json.dump(upload_log, f, indent=4)
    else:
        print("No file name was given")


################################################################################## 
# Run the program 
################################################################################## 

def run(sales_data: pd.DataFrame = None, input: bool = False):
    """
    Main function to process sales data, train forecasting models, predict machine and component demand,
    compute lead times, and simulate inventory levels.

    Parameters:
        sales_data: pd.DataFrame containing initial sales data for processing
        input: Indicates whether the function is activated by a new input file in the Taipy app (True),
                or run outside the app (False).

    Returns:
        None
    """

    if input == True:
        print("run() function has been activated")
        print("sales_data:")
        print(sales_data.head())

    aggregation_methods = ["month", "week"]

    now = datetime.now().isoformat(timespec='seconds')
    now = now.replace(":", "-").replace("T", "_")

    for i in aggregation_methods:
        # Define and load data
        sales_data, component_data, inventory_data, machine_dict, machine_types, aggregation_method, n_lags, varegrupper, non_representative_years, forecast_periods = take_inputs(sales_data=sales_data, aggregation_method=i)

        original_sales = prepare_sales_data(data=sales_data,
                                            machine_dict = machine_dict,
                                            aggregation_method=i,
                                            non_representative_years=non_representative_years)

        if 'unknown model' in original_sales.columns:
            original_sales = original_sales.drop(columns=['unknown model'])

        lagged_sales = {}
        selected_lags = {}

        for key in machine_dict.keys():
            lagged_sales[key], selected_lags[key] = create_dataset_with_lagged_features(data=original_sales[[key]], 
                                                                n_lags=n_lags, 
                                                                aggregation_method=i)
        
        print("----------------")
        print("Data generated")
        print("Now, training models")
        print("----------------")

        # Train models
        model_dict = {key: train_forecasting_model(data=lagged_sales[key], 
                                                            aggregation_method=i) 
                                                            for key in lagged_sales.keys()}

        for key in model_dict.keys():
            plot_predictions(model_dict[key], filename=f'plot_{aggregation_method}_{key}', key=key, input=input, timestamp=str(now))

        # Prepare data for prediction
        prediction_data_machine = {key:prepare_machine_data_for_prediction(data=original_sales[[key]],
                                                                        aggregation_method=i)
                                                            for key in model_dict.keys()}

        prediction_data_component = prepare_component_data_for_prediction(data=component_data, machine_dict=machine_dict)
        
        # Predict machine demand
        predicted_machine_demand = {key: predict_machine_demand(model_dict=model_dict[key], 
                                                        data_for_prediction=prediction_data_machine[key], 
                                                        pacf_lags=selected_lags[key], 
                                                        forecast_unit=aggregation_method, 
                                                        forecast_periods=forecast_periods)
                                                        for key in prediction_data_machine.keys()}
        
        predicted_machine_demand_df = pd.DataFrame()

        for key in predicted_machine_demand.keys():
            df = predicted_machine_demand[key].copy()
            df.columns = [key]
            predicted_machine_demand_df = pd.concat([predicted_machine_demand_df, df], axis=1)
        
        predicted_machine_demand_df = predicted_machine_demand_df.sort_index()

        print("-----------")
        print("Predicted machine demand df")
        print(predicted_machine_demand_df.head())
        print("-----------")


        # Predict component demand
        historical_component_demand_df = convert_machine_to_component_demand(machine_demand_df=original_sales, 
                                                                            machine_types=machine_types)
        
        predicted_component_demand_df = convert_machine_to_component_demand(machine_demand_df=predicted_machine_demand_df, 
                                                                            machine_types=machine_types)

        print("-----------")
        print("Predicted component demand df")
        print(predicted_component_demand_df.head())
        print("-----------")

        # Compute lead_times
        lead_times = compute_lead_time(data=inventory_data, 
                                    varegrupper=varegrupper, 
                                    time_unit_mapping=None,
                                    aggregation_method=aggregation_method)
        
        lead_times_avr, lead_times_max = calculate_lead_times(lead_times=lead_times, aggregation_method=aggregation_method)
        
        # Define variables for inventory simulation
        component_types = {column: None for column in predicted_component_demand_df.columns}
        component_max_demand = {component: int(predicted_component_demand_df[component].max()) for component in component_types.keys()}
        component_avr_demand = {component: int(predicted_component_demand_df[component].mean()) for component in component_types.keys()}
        component_safety_stock = {component: safety_stock(max_demand=component_max_demand[component], max_lead_time=lead_times_max[component], avr_demand=component_avr_demand[component], avr_lead_time=lead_times_avr[component]) for component in component_types.keys()}
        component_initial_inventory = {component: max([component_safety_stock[component], component_max_demand[component]]) for component in component_types.keys()} 

        inventory_history, order_history, reorder_points = simulate_inventory_fixed(component_demand_df=predicted_component_demand_df,
                                                                                    component_types=component_types,
                                                                                    initial_inventory=component_initial_inventory,
                                                                                    lead_times=lead_times_avr,
                                                                                    safety_stock=component_safety_stock)
        
        inventory_df = pd.DataFrame(inventory_history, index=predicted_component_demand_df.index)
        reorder_df = pd.DataFrame(reorder_points, index=predicted_component_demand_df.index)


        if input == True:
            base_path = os.path.dirname(__file__) 
            output_dir = os.path.join(base_path, *["DataInputs", "DataTrained"])
            os.makedirs(output_dir, exist_ok=True)

            file_name = f"data_{aggregation_method}_{now}.pkl"
            full_path = os.path.join(output_dir, file_name)

            with open(full_path, "wb") as f:
                pickle.dump((original_sales,
                            predicted_machine_demand_df, 
                            historical_component_demand_df,
                            predicted_component_demand_df, 
                            inventory_df, 
                            reorder_df,
                            order_history,
                            component_types,
                            component_initial_inventory,
                            lead_times_avr,
                            component_safety_stock), f)
                
            update_upload_log(file_name = file_name, timestamp = now, path = output_dir, aggregation_method=i)

        else:
            base_path = os.path.dirname(__file__) 
            output_dir = os.path.join(base_path, *["DataDefault", "DataTrained"])
            os.makedirs(output_dir, exist_ok=True)

            file_name = f"data_{aggregation_method}.pkl"
            full_path = os.path.join(output_dir, file_name)

            with open(full_path, "wb") as f:
                pickle.dump((original_sales,
                            predicted_machine_demand_df, 
                            historical_component_demand_df,
                            predicted_component_demand_df, 
                            inventory_df, 
                            reorder_df,
                            order_history,
                            component_types,
                            component_initial_inventory,
                            lead_times_avr,
                            component_safety_stock), f)

            update_upload_log(file_name = file_name, timestamp = now, path = output_dir, aggregation_method=i)

if __name__ == "__main__":
    run()

