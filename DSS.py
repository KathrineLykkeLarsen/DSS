#############################################################
# Import libraries
#############################################################

import taipy as tp
import taipy.gui.builder as tgb
from taipy.gui import Icon, notify
from taipy import Config
import pandas as pd
from datetime import timedelta, datetime
import json
import os
from BackendFunctionsGeneral import simulate_inventory_fixed
from BackendFunctionsApp import *

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

#############################################################
# Load data
#############################################################

try:
    # Read the log file
    base_path = os.path.dirname(__file__) 
    relative_path = ["Logs", "upload_log.json"]
    log_file = os.path.join(base_path, *relative_path)

    with open(log_file, "r") as f:
        log_data = json.load(f)
    
    # Find newest entry
    month_entries = [entry for entry in log_data if entry["aggregation_method"] == "month"]
    week_entries = [entry for entry in log_data if entry["aggregation_method"] == "week"]

    latest_entry_month = max(month_entries,
                       key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))
    latest_entry_week = max(week_entries,
                       key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))

    path = os.path.join(base_path, "DataInputs", "DataTrained")
    file_name_month = latest_entry_month["filename"]
    full_path_month = os.path.join(path, file_name_month)
    file_name_week = latest_entry_week["filename"]
    full_path_week = os.path.join(path, file_name_week)

    
    # Monthly data
    original_sales_month, predicted_machine_demand_df_month, historical_component_demand_df_month, predicted_component_demand_df_month, inventory_df_month, reorder_df_month, order_history_month, component_types_month, component_initial_inventory_month, lead_times_avr_month, component_safety_stock_month = pd.read_pickle((full_path_month))
    inventory_df_month = inventory_df_month.fillna(0).round(0).astype(int)
    reorder_df_month = reorder_df_month.fillna(0).round(0).astype(int)

    data_machine_month = pd.concat([original_sales_month, predicted_machine_demand_df_month])
    data_component_month = pd.concat([historical_component_demand_df_month, predicted_component_demand_df_month])

    # Weekly data
    original_sales_week, predicted_machine_demand_df_week, historical_component_demand_df_week, predicted_component_demand_df_week, inventory_df_week, reorder_df_week, order_history_week, component_types_week, component_initial_inventory_week, lead_times_avr_week, component_safety_stock_week = pd.read_pickle((full_path_week))
    inventory_df_week = inventory_df_week.fillna(0).round(0).astype(int)
    reorder_df_week = reorder_df_week.fillna(0).round(0).astype(int)

    data_machine_week = pd.concat([original_sales_week, predicted_machine_demand_df_week])
    data_component_week = pd.concat([historical_component_demand_df_week, predicted_component_demand_df_week])

except:
    base_path = os.path.dirname(__file__) 
    path = os.path.join(base_path, "DataDefault", "DataTrained")
    full_path_month = os.path.join(path, "data_month.pkl")
    full_path_week = os.path.join(path, "data_week.pkl")

    # Monthly data
    original_sales_month, predicted_machine_demand_df_month, historical_component_demand_df_month, predicted_component_demand_df_month, inventory_df_month, reorder_df_month, order_history_month, component_types_month, component_initial_inventory_month, lead_times_avr_month, component_safety_stock_month = pd.read_pickle((full_path_month))
    inventory_df_month = inventory_df_month.fillna(0).round(0).astype(int)
    reorder_df_month = reorder_df_month.fillna(0).round(0).astype(int)

    data_machine_month = pd.concat([original_sales_month, predicted_machine_demand_df_month])
    data_component_month = pd.concat([historical_component_demand_df_month, predicted_component_demand_df_month])

    # Weekly data
    original_sales_week, predicted_machine_demand_df_week, historical_component_demand_df_week, predicted_component_demand_df_week, inventory_df_week, reorder_df_week, order_history_week, component_types_week, component_initial_inventory_week, lead_times_avr_week, component_safety_stock_week = pd.read_pickle((full_path_week))
    inventory_df_week = inventory_df_week.fillna(0).round(0).astype(int)
    reorder_df_week = reorder_df_week.fillna(0).round(0).astype(int)

    data_machine_week = pd.concat([original_sales_week, predicted_machine_demand_df_week])
    data_component_week = pd.concat([historical_component_demand_df_week, predicted_component_demand_df_week])

#############################################################
# Define initial values
#############################################################

# Jeros logo
base_path = os.path.dirname(__file__) 
relative_path = ["Icons", "Jeros_logo.png"]
jeros_logo_file = os.path.join(base_path, *relative_path)


# Initial value for file input
path = None
new_data = None

# Initial dates
date = data_machine_month.index.max()-timedelta(15*365.25/12)
start_date = data_machine_month.index.min()
end_date = data_machine_month.index.max()

# Initial machine and component values
machine_names = data_machine_month.columns.values.tolist()
machine = machine_names.copy()
machine_names = [(name, Icon("Icons/" + name + ".png", name)) for name in machine_names]

machine_types = {
        "kassevasker": ["Slanger", "Doseringspumpe", "Vaskepumpe", "Styring", "Varmelegeme"],
        "opvasker": ["Slanger", "Doseringspumpe", "Vaskepumpe", "Skyllepumpe", "Sæbepumpe", "Drænpumpe", "Styring"],
        "pladerenser_type9020": ["Doseringspumpe", "Styring", "Børster", "Valser", "Olieakser"],
        "pladerenser_type9029":["Vakuumpumpe", "Styring", "Børster", "Valser", "Olieakser"],
        "pladerenser_type3":["Styring", "Børster", "Valser", "Olieakser"]}
component_names = list({
                        comp
                        for m in machine
                        for comp in machine_types.get(m, [])})
component = component_names.copy()
inventory_component = component_names.copy()[0]

# Initial dropdown values
agg_methods = ["Month", "Week"]
agg_method = agg_methods[0]

show_demand_options = ["No", "Yes"]
show_demand_option = ["Yes"]

show_scenarios = ["No", "Yes"]
show_scenario = show_scenarios[0]

scenario_values = ["Scenario 1", "Scenario 2"]
scenario_value = scenario_values[0]

# Initial graph data and table data
graph_data_machine = pd.DataFrame()
graph_data_component = pd.DataFrame()
graph_data_inventory = pd.DataFrame()
table_data_inventory = pd.DataFrame()
table_data_scenario1 = pd.DataFrame(columns=["Date", "Component", "Reorder Point", "Lead Time", "Safety Stock", "Demand", "Inventory", "Order Quantity"])
table_data_scenario2 = pd.DataFrame(columns=["Date", "Component", "Reorder Point", "Lead Time", "Safety Stock", "Demand", "Inventory", "Order Quantity"])

# Initial figures
figure_machine = None
figure_component = None
figure_inventory = None

# Initial input values for scenario analysis
input_inv1 = None
input_inv2 = None
input_ss1 = None
input_ss2 = None
input_lt1 = None
input_lt2 = None
override_order_qty1 = None
override_order_qty2 = None
override_order_qty_period1 = None
override_order_qty_period2 = None

#############################################################
# Define button functionality
#############################################################
def adjust_inventory(state):
    """
    Simulates inventory for a selected component under a given scenario and updates the scenario table.

    Parameters:
        state: An object containing input values for scenario configuration, including inventory parameters,
               aggregation level (week/month), and selected component.

    Returns:
        None
    """

    agg_method = state.agg_method.lower()
    comp = state.inventory_component
    scenario_val = state.scenario_value

    if scenario_val == "Scenario 1":
        input_inv = state.input_inv1
        input_ss = state.input_ss1
        input_lt = state.input_lt1
        override_order_qty = state.override_order_qty1
        override_order_qty_period = state.override_order_qty_period1
    
    elif scenario_val == "Scenario 2":
        input_inv = state.input_inv2
        input_ss = state.input_ss2
        input_lt = state.input_lt2
        override_order_qty = state.override_order_qty1
        override_order_qty_period = state.override_order_qty_period1

    else:
        print("Scenario value unrecognized")


    input_values = {
        "Initial Inventory": input_inv,
        "Safety Stock": input_ss,
        "Lead Time": input_lt,
        "Override Order Quantity": override_order_qty,
        "Override Order Period": override_order_qty_period,
    }

    for label, val in input_values.items():
        try:
            if val is not None and float(val) < 0:
                notify(state, "warning", f"{label} must be non-negative")
                # Create empty dataframe
                df = pd.DataFrame()
                state.table_data_scenario1 = df.copy()
                state.table_data_scenario2 = df.copy()
                return
        except ValueError:
            notify(state, "warning", f"{label} must be a numerical value")
            # Create empty dataframe
            df = pd.DataFrame()
            state.table_data_scenario1 = df.copy()
            state.table_data_scenario2 = df.copy()
            return
    

    if agg_method == "month":
        override_qty = safe_float_to_int(override_order_qty, None)
        override_period = (int(override_order_qty_period)-1
                           if str(override_order_qty_period).isdigit() and 1 <= int(override_order_qty_period) <= 3
                           else None)

        predicted_component_demand_df = predicted_component_demand_df_month.copy()
        component_types_month_scenario = component_types_month.copy()

        component_initial_inventory_month_scenario = component_initial_inventory_month.copy()
        component_initial_inventory_month_scenario[comp] = safe_float_to_int(input_inv, component_initial_inventory_month[comp])

        component_safety_stock_scenario = component_safety_stock_month.copy()
        component_safety_stock_scenario[comp] = safe_float_to_int(input_ss, component_safety_stock_month[comp])

        lead_times_avr_scenario = lead_times_avr_month.copy()
        lead_times_avr_scenario[comp] = safe_float_to_int(input_lt, lead_times_avr_month[comp], scale=1/30)
  
        inventory_month_scenario, order_history, reorder_month_scenario = simulate_inventory_fixed(component_demand_df=predicted_component_demand_df,
                                                                                        component_types=component_types_month_scenario,
                                                                                        initial_inventory=component_initial_inventory_month_scenario,
                                                                                        lead_times=lead_times_avr_scenario,
                                                                                        safety_stock=component_safety_stock_scenario,
                                                                                        override_order_qty=override_qty,
                                                                                        override_period=override_period)

        inventory_df = pd.DataFrame(inventory_month_scenario, index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)
        reorder_df = pd.DataFrame(reorder_month_scenario, index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)

    elif agg_method == "week":
        override_qty = safe_float_to_int(override_order_qty, None)
        override_period = (int(override_order_qty_period)-1
                           if str(override_order_qty_period).isdigit() and 1 <= int(override_order_qty_period) <= 3
                           else None)

        predicted_component_demand_df = predicted_component_demand_df_week.copy()
        component_types_week_scenario = component_types_week.copy()

        component_initial_inventory_week_scenario = component_initial_inventory_week.copy()
        component_initial_inventory_week_scenario[comp] = safe_float_to_int(input_inv, component_initial_inventory_week[comp])

        component_safety_stock_scenario = component_safety_stock_week.copy()
        component_safety_stock_scenario[comp] = safe_float_to_int(input_ss, component_safety_stock_week[comp])

        lead_times_avr_scenario = lead_times_avr_week.copy()
        lead_times_avr_scenario[comp] = safe_float_to_int(input_lt, lead_times_avr_week[comp], scale=1/4)
  
        inventory_week_scenario, order_history, reorder_week_scenario = simulate_inventory_fixed(component_demand_df=predicted_component_demand_df,
                                                                                        component_types=component_types_week_scenario,
                                                                                        initial_inventory=component_initial_inventory_week_scenario,
                                                                                        lead_times=lead_times_avr_scenario,
                                                                                        safety_stock=component_safety_stock_scenario,
                                                                                        override_order_qty=override_qty,
                                                                                        override_period=override_period)

        inventory_df = pd.DataFrame(inventory_week_scenario, index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)
        reorder_df = pd.DataFrame(reorder_week_scenario, index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)

    else:
        print("Aggregation method unrecognized")

    # Create new dataset for table
    if isinstance(comp, str):
        inventory_component = [comp]
                
    if reorder_df.empty or inventory_df.empty:
        # Create empty dataframe
        df = pd.DataFrame()
    else:
        graph_data = {i: {"reorder_point":reorder_df[i],
                          "lead_time": lead_times_avr_scenario[i],
                          "safety_stock": component_safety_stock_scenario[i],
                          "demand": predicted_component_demand_df[i],
                          "inventory": inventory_df[i],
                          "quantity": pd.DataFrame(order_history[i], columns=["Order Month", "Order Quantity"], index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)}
                          for i in inventory_component}
        records = []
        for inventory_component, data in graph_data.items():
            dates = data['reorder_point'].index
            for date in dates:
                record = {
                    'Date': date,
                    'Component': inventory_component,
                    'Reorder Point': data['reorder_point'].loc[date],
                    'Lead Time': int(data['lead_time']),
                    'Safety Stock': int(data['safety_stock']),
                    'Demand': data['demand'].loc[date],
                    'Inventory': data['inventory'].loc[date]}
                

                order_quantity = data['quantity'].loc[date, 'Order Quantity']
                record['Order Quantity'] = order_quantity

                records.append(record)

        # Convert to pandas dataframe
        df = pd.DataFrame(records)
        df['Date'] = df["Date"].dt.strftime("%b %Y")
        
        for col in ["Reorder Point", "Demand", "Lead Time", "Safety Stock", "Inventory", "Order Quantity"]:
            df[col] = df[col].astype(int).astype(str)

        if (df["Inventory"].astype(int) <= 0).any():
            notify(state, "warning", "Inventory is zero or negative in one or more periods. " \
            "                         This may lead to stockouts. Consider adjusting your order quantities accordingly.")
        

    # Update scenario table 
    if scenario_val == "Scenario 1":
        state.table_data_scenario1 = df.copy()
    elif scenario_val == "Scenario 2":
        state.table_data_scenario2 = df.copy()
    else:
        print("Scenario value unrecognized")

#############################################################
# Frontend functions
#############################################################

def build_component_names(machine):
    """
    Builds a list of unique component names for a given machine or list of machines.

    Parameters:
        machine: A single machine name (str) or a list of machine names.

    Returns:
        list: A list of unique component names associated with the machine(s).
    """

    if type(machine)==list:
        component_names = []
        for i in machine:
            [component_names.append(x) for x in machine_types[i] if x not in component_names]
    else:
        component_names = machine_types[machine]
    return component_names


def build_graph_data_machine(start_date, machine, agg_method):
    """
    Returns filtered machine demand data from a selected dataset based on aggregation method.

    Parameters:
        start_date: The start date (str or datetime) to filter from.
        machine: A single machine name (str) or a list of machine names.
        agg_method: Aggregation method used to select the dataset ('month' or 'week').

    Returns:
        pandas.DataFrame: Filtered demand data for the selected machine(s).
    """

    agg_method = agg_method.lower()
    if agg_method == "month":
        data_machine = data_machine_month.copy()
    elif agg_method == "week":
        data_machine = data_machine_week.copy()

    if isinstance(machine, str):
        machine = [machine]
    
    if len(machine) > 0:
        graph_data_machine = data_machine[machine][
                            data_machine.index >= pd.to_datetime(start_date)]
    else:
        # Return empty dataframe
        graph_data_machine = pd.DataFrame()
    return graph_data_machine


def build_graph_data_component(start_date, component_names, component, agg_method):
    """
    Filters component-level demand data based on selected components and aggregation method.

    Parameters:
        start_date: The start date (str or datetime) to filter from.
        component_names: A list of valid component names for validation.
        component: A single component name (str) or a list of component names to include.
        agg_method: Aggregation method used to select the dataset ('month' or 'week').

    Returns:
        pandas.DataFrame: Filtered demand data for the selected component(s).
    """

    agg_method = agg_method.lower()
    if agg_method == "month":
        data_component = data_component_month.copy()
    elif agg_method == "week":
        data_component = data_component_week.copy()

    if isinstance(component, str):
        component = [component]
    
    if len(component) > 0:
        component_to_plot = [comp for comp in component if comp in component_names]
        graph_data_component = data_component[component_to_plot][
                                                    data_component.index >= pd.to_datetime(start_date)]
    else:
        # Return empty dataframe
        graph_data_component = pd.DataFrame()
    return graph_data_component


def build_graph_data_inventory(component_names, inventory_component, show_demand_option, agg_method):
    """
    Builds inventory graph data for a selected component based on aggregation method and demand display option.

    Parameters:
        component_names: List of valid component names for validation.
        inventory_component: Single component name or list of component names to include.
        show_demand_option: 'Yes' or 'No' indicating whether to include predicted demand in the output.
        agg_method: Aggregation method used to select dataset ('month' or 'week').

    Returns:
        dict or None: Dictionary with inventory, quantity, and optionally demand data per component. Returns None if input is invalid.
    """
    
    agg_method = agg_method.lower()
    if agg_method == "month":
        order_history = order_history_month.copy()
        inventory_df = inventory_df_month.copy()
        predicted_component_demand_df = predicted_component_demand_df_month.copy()
    elif agg_method == "week":
        inventory_df = inventory_df_week.copy()
        order_history = order_history_week.copy()
        predicted_component_demand_df = predicted_component_demand_df_week.copy()

    if isinstance(inventory_component, str):
        inventory_component = [inventory_component]
    
    if len(inventory_component) > 0:
        component_to_plot = [comp for comp in inventory_component if comp in component_names]

        if isinstance(show_demand_option, list):
            show_demand_option = show_demand_option[0]
        
        if show_demand_option == "No":
            graph_data = {i: {"inventory": inventory_df[i],
                            "quantity": pd.DataFrame(order_history[i], columns=["Order Month", "Order Quantity"], index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)}
                        for i in component_to_plot}
        elif show_demand_option == "Yes":
            graph_data = {i: {"demand": predicted_component_demand_df[i],
                            "inventory": inventory_df[i],
                            "quantity": pd.DataFrame(order_history[i], columns=["Order Month", "Order Quantity"], index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)}
                        for i in component_to_plot}
        else:
            graph_data = None
    else:
        graph_data = None
    return graph_data


def build_table_data_inventory(component_names, inventory_component, agg_method):
    """
    Builds a table of inventory metrics for a selected component based on aggregation method.

    Parameters:
        component_names: List of valid component names for validation.
        inventory_component: Single component name or list of component names to include in the table.
        agg_method: Aggregation method used to select dataset ('month' or 'week').

    Returns:
        pd.DataFrame: DataFrame with inventory metrics including reorder point, demand, safety stock,
                      lead time, and order quantity per timeperiod. Returns empty DataFrame if input is invalid.
    """

    agg_method = agg_method.lower()
    if agg_method == "month":
        reorder_df = reorder_df_month.copy()
        inventory_df = inventory_df_month.copy()
        predicted_component_demand_df = predicted_component_demand_df_month.copy()
        order_history = order_history_month.copy()
        lead_times = lead_times_avr_month.copy()
        component_safety_stock = component_safety_stock_month.copy()
    elif agg_method == "week":
        inventory_df = inventory_df_week.copy()
        reorder_df = reorder_df_week.copy()
        predicted_component_demand_df = predicted_component_demand_df_week.copy()
        order_history = order_history_week.copy()
        lead_times = lead_times_avr_week.copy()
        component_safety_stock = component_safety_stock_week.copy()

    if isinstance(inventory_component, str):
        inventory_component = [inventory_component]
    
    if len(component) > 0:
        component_to_plot = [comp for comp in inventory_component if comp in component_names]
        
        graph_data = {i: {"reorder_point":reorder_df[i],
                          "lead_time": lead_times[i],
                          "safety_stock": component_safety_stock[i],
                          "demand": predicted_component_demand_df[i],
                          "inventory": inventory_df[i],
                          "quantity": pd.DataFrame(order_history[i], columns=["Order Month", "Order Quantity"], index=predicted_component_demand_df.index).fillna(0).round(0).astype(int)}
                          for i in component_to_plot}

        records = []

        for inventory_component, data in graph_data.items():
            dates = data['reorder_point'].index

            for date in dates:
                record = {
                    'Date': date,
                    'Component': inventory_component,
                    'Reorder Point': data['reorder_point'].loc[date],
                    'Lead Time': int(data['lead_time']),
                    'Safety Stock': int(data['safety_stock']),
                    'Demand': data['demand'].loc[date],
                    'Inventory': data['inventory'].loc[date],
                }
                order_quantity = data['quantity'].loc[date, 'Order Quantity']
                record['Order Quantity'] = order_quantity

                records.append(record)

        # Convert to pandas dataframe
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # Return empty dataframe
        df = pd.DataFrame()
    return df


#############################################################
# Create app
#############################################################

with tgb.Page() as page:
    with tgb.part("text-center"):
        with tgb.layout("10 80 10"):
            tgb.image(jeros_logo_file, width="10vw")

            tgb.text(
                    "Decision Support Tool for Jeros A/S",
                    mode='md',
                    class_name="h1")
            
            tgb.file_selector(content='{path}', label="Upload dataset", on_action=load_xlsx_file, class_name="fullwidth", extensions='xlsx')
        
        # Machine demand
        tgb.text(
                "Machine demand",
                mode='md',
                class_name="h2")
        with tgb.layout("50 50"):
            tgb.date("{date}", 
                     min="{start_date}", 
                     max="{end_date}", 
                     label="Display data starting from", 
                     class_name="fullwidth")

            tgb.selector(label="Aggregate data by",
                            class_name="fullwidth",
                            value="{agg_method}",
                            lov="{agg_methods}",
                            dropdown=True,
                            value_by_id=True)
            
        tgb.selector(label="Filter machines",
                            class_name="fullwidth",
                            value="{machine}",
                            lov="{machine_names}",
                            dropdown=True,
                            value_by_id=True,
                            multiple=True)
        
        tgb.chart(figure="{figure_machine}")

        # Component demand
        tgb.text(
                "Component demand",
                mode='md',
                class_name="h2")
        
        tgb.selector(label="Filter components",
                    class_name="fullwidth",
                    value="{component}",
                    lov="{component_names}",
                    dropdown=True,
                    value_by_id=True,
                    multiple=True)
        
        tgb.chart(figure="{figure_component}")
        
        with tgb.part("text-center", 
                      render="{state.component is not None and state.component != ''}"):
            tgb.text(
                "Inventory management",
                mode='md',
                class_name="h2")
            
            with tgb.part("text-center"):
                with tgb.layout("30 70"):
                    with tgb.part("text-center"):
                        tgb.selector(label="Filter components in inventory overview",
                                class_name="fullwidth",
                                value="{inventory_component}",
                                lov="{component_names}",
                                dropdown=True,
                                value_by_id=True,
                                multiple=False)
                        
                        tgb.selector(label="Show predicted demand in inventory overview",
                                class_name="fullwidth",
                                value="{show_demand_option}",
                                lov="{show_demand_options}",
                                dropdown=True,
                                value_by_id=True,
                                multiple=False)

                    tgb.chart(figure="{figure_inventory}")

                with tgb.layout("30 70"):
                    tgb.selector(label="Compare scenarios",
                                class_name="fullwidth",
                                value="{show_scenario}",
                                lov="{show_scenarios}",
                                dropdown=True,
                                value_by_id=True,
                                multiple=False)

                    tgb.table(data="{table_data_inventory}", 
                                page_size=30, 
                                page_size_options=[30, 60, 90], 
                                date_format="MMM yyyy")
        
        # Scenario analysis
        with tgb.part("text-center", render="{show_scenario == 'Yes'}"):
            tgb.text(
                "Scenario Analysis",
                mode='md',
                class_name="h2")

            with tgb.layout("30 70"):
                with tgb.part("text-center"):
                    tgb.selector(label="Change scenario",
                                value="{scenario_value}",
                                lov="{scenario_values}",
                                dropdown=True,
                                value_by_id=True,
                                multiple=False)
                with tgb.part("text-center"):   
                    tgb.button(label="Reset inventory simulation", on_action=reset_simulation, class_name="fullwidth")

            # Scenario analysis part 1
            with tgb.part("text-center", render="{scenario_value == 'Scenario 1'}"):
                with tgb.layout("30 70"):
                    with tgb.part("text-center"):
                        with tgb.layout("50 50"):
                            tgb.input(value="{input_inv1}", label="Initial inventory level", class_name="fullwidth")
                            tgb.input(value="{input_lt1}", label="Current lead time (days)", class_name="fullwidth")
                        tgb.input(value="{input_ss1}", label="Preferred safety stock (number of components)", class_name="fullwidth")
                        with tgb.layout("50 50"):        
                            tgb.input(value="{override_order_qty1}", label="Order quantity override", class_name="fullwidth")
                            tgb.input(value="{override_order_qty_period1}", label="Which period to apply it to", class_name="fullwidth")
                        tgb.button(label="Simulate inventory", on_action=adjust_inventory, class_name="fullwidth")
                    with tgb.part("text-center"):
                        tgb.table(data="{table_data_scenario1}", 
                            page_size=30, 
                            page_size_options=[30, 60, 90],
                            date_format="MMM yyyy")

            # Scenario analysis part 2
            with tgb.part("text-center", render="{scenario_value == 'Scenario 2'}"):
                with tgb.layout("30 70"):
                    with tgb.part("text-center"):
                        with tgb.layout("50 50"):
                            tgb.input(value="{input_inv2}", label="Initial inventory level", class_name="fullwidth")
                            tgb.input(value="{input_lt2}", label="Current lead time (days)", class_name="fullwidth")
                        tgb.input(value="{input_ss2}", label="Preferred safety stock (number of components)", class_name="fullwidth")
                        with tgb.layout("50 50"):        
                            tgb.input(value="{override_order_qty2}", label="Order quantity override", class_name="fullwidth")
                            tgb.input(value="{override_order_qty_period2}", label="Which period to apply it to", class_name="fullwidth")
                        tgb.button(label="Simulate inventory", on_action=adjust_inventory, class_name="fullwidth")
                    with tgb.part("text-center"):
                        tgb.table(data="{table_data_scenario2}", 
                            page_size=30, 
                            page_size_options=[30, 60, 90],
                            date_format="MMM yyyy")

#############################################################
# Scenarios
#############################################################

# Input
machine_cfg = Config.configure_data_node(
    id="machine")
# Output
component_names_cfg = Config.configure_data_node(
    id="component_names")

# Input
date_cfg = Config.configure_data_node(
    id="date")
# Output
graph_data_machine_cfg = Config.configure_data_node(
    id="graph_data_machine") 

# Input
component_cfg = Config.configure_data_node(
    id="component")
# Output
graph_data_component_cfg = Config.configure_data_node(
    id="graph_data_component")

# Input
inventory_input1_cfg = Config.configure_data_node(
    id="inventory_input1")
inventory_input2_cfg = Config.configure_data_node(
    id="inventory_input2")
inventory_input3_cfg = Config.configure_data_node(
    id="inventory_input3")
override_order_qty_cfg = Config.configure_data_node(
    id="override_order_qty")
override_order_qty_period_cfg = Config.configure_data_node(
    id="override_order_qty_period")

# Input
inventory_component_cfg = Config.configure_data_node(
    id="inventory_component")
# Input
show_demand_option_cfg = Config.configure_data_node(
    id="show_demand_option")


#Output
graph_data_inventory_cfg = Config.configure_data_node(
    id="graph_data_inventory")

# Output
table_data_inventory_cfg = Config.configure_data_node(
    id="table_data_inventory")

# Input
agg_method_cfg = Config.configure_data_node(
    id="agg_method")

# Output
table_data_scenario_cfg = Config.configure_data_node(
    id="table_data_scenario")


# Task: Choose machine, load possible components in that machine
build_component_names_cfg = Config.configure_task(
                                input=machine_cfg,
                                output=component_names_cfg,
                                function=build_component_names,
                                id="build_component_names",
                                skippable=True)

# Task: Plot machine demand
build_graph_data_machine_cfg = Config.configure_task(
                                input=[date_cfg, machine_cfg, agg_method_cfg],
                                output=graph_data_machine_cfg,
                                function=build_graph_data_machine,
                                id="build_graph_data_machine",
                                skippable=True)

# Task: Plot component demand
build_graph_data_component_cfg = Config.configure_task(
                                input=[date_cfg, component_names_cfg, component_cfg, agg_method_cfg],
                                output=graph_data_component_cfg,
                                function=build_graph_data_component,
                                id="build_graph_data_component",
                                skippable=True)

# Task: Plot inventory overview
build_graph_data_inventory_cfg = Config.configure_task(
                                input=[component_names_cfg, inventory_component_cfg, show_demand_option_cfg, agg_method_cfg],
                                output=graph_data_inventory_cfg,
                                function=build_graph_data_inventory,
                                id="build_graph_data_inventory",
                                skippable=True)

# Task: Plot inventory overview
build_table_data_inventory_cfg = Config.configure_task(
                                input=[component_names_cfg, inventory_component_cfg, agg_method_cfg],
                                output=table_data_inventory_cfg,
                                function=build_table_data_inventory,
                                id="build_table_data_inventory",
                                skippable=True)

# Create scenario
scenario_cfg = Config.configure_scenario(
                    task_configs=[build_component_names_cfg,
                                  build_graph_data_machine_cfg,
                                  build_graph_data_component_cfg,
                                  build_graph_data_inventory_cfg,
                                  build_table_data_inventory_cfg],
                    additional_data_node_configs=[inventory_input1_cfg, inventory_input2_cfg, inventory_input3_cfg, override_order_qty_cfg, override_order_qty_period_cfg, table_data_scenario_cfg],
                    id="scenario")

#############################################################
# Backend functions
#############################################################

def on_init(state):
    """
    Initializes the app state by synchronizing scenario inputs and reading output data.

    Parameters:
        state: The app state object used to read and write scenario parameters and results.

    Side Effects:
        Updates the state with scenario parameters and loads initial data for graphs and tables.
    """

    state.scenario.date.write(state.date)
    state.scenario.machine.write(state.machine)
    state.scenario.component.write(state.component)
    state.scenario.inventory_component.write(state.inventory_component)
    state.scenario.show_demand_option.write(state.show_demand_option)
    state.scenario.agg_method.write(state.agg_method)
    state.scenario.submit(wait=True)
    state.component_names = state.scenario.component_names.read()
    state.graph_data_machine = state.scenario.graph_data_machine.read()
    state.graph_data_component = state.scenario.graph_data_component.read()
    state.graph_data_inventory = state.scenario.graph_data_inventory.read()
    state.table_data_inventory = state.scenario.table_data_inventory.read()

# Fetch selection
def on_change(state, name):
    """
    Handles user interface (UI) changes by updating scenario inputs, triggering computation, and updating graphs/tables.

    Parameters:
        state: The app state object containing current UI and scenario values.
        name: The name of the input that changed.

    Side Effects:
        Updates the state with new figures, table data, and resets scenario analysis parts when relevant.
    """

    if name == "machine" or name == "component":
        # Write
        state.scenario.machine.write(state.machine)
        state.scenario.component.write(state.component)
        state.scenario.inventory_component.write(state.inventory_component)
        # Submit
        state.scenario.submit(wait=True)
        # Read
        state.component_names = state.scenario.component_names.read()
        state.graph_data_machine = state.scenario.graph_data_machine.read()
        state.graph_data_component = state.scenario.graph_data_component.read()
        state.graph_data_inventory = state.scenario.graph_data_inventory.read()
        state.table_data_inventory = state.scenario.table_data_inventory.read()
        
        # Reset simulation
        reset_simulation(state, both=True)

    if name == "date":
        # Write
        state.scenario.date.write(state.date)
        # Submit
        state.scenario.submit(wait=True)
        # Read
        state.graph_data_machine = state.scenario.graph_data_machine.read()
        state.graph_data_component = state.scenario.graph_data_component.read()

    if name == "graph_data_machine":
        state.figure_machine = plot_machine(state.graph_data_machine)
    
    if name == "graph_data_component":
        state.figure_component = plot_component(state.graph_data_component)

    if name == "show_demand_option":
        # Write
        state.scenario.show_demand_option.write(state.show_demand_option)
        # Submit
        state.scenario.submit(wait=True)
        # Read
        state.graph_data_inventory = state.scenario.graph_data_inventory.read()

    if name == "inventory_component":
        # Write
        state.scenario.inventory_component.write(state.inventory_component)
        # Submit
        state.scenario.submit(wait=True)
        # Read
        state.graph_data_inventory = state.scenario.graph_data_inventory.read()
        state.table_data_inventory = state.scenario.table_data_inventory.read()

    if name == "agg_method":
        # Write
        state.scenario.agg_method.write(state.agg_method)
        # Submit
        state.scenario.submit(wait=True)
        # Read
        state.graph_data_machine = state.scenario.graph_data_machine.read()
        state.graph_data_component = state.scenario.graph_data_component.read()
        state.graph_data_inventory = state.scenario.graph_data_inventory.read()
        state.table_data_inventory = state.scenario.table_data_inventory.read()

        # Reset simulation
        reset_simulation(state, both=True)

    if name == "graph_data_inventory":
        state.figure_inventory = plot_inventory(graph_data=state.graph_data_inventory)

#############################################################
# Run the app
#############################################################
if __name__ == "__main__":
    tp.Orchestrator().run()
    scenario = tp.create_scenario(scenario_cfg)
    gui = tp.Gui(page)
    gui.run(
            title="Decision Support Tool",
            use_reloader=True)
    