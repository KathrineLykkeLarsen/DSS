#############################################################
# Import libraries
#############################################################

import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import math
from BackendFunctionsGeneral import run
from BackendFunctionsApp import *

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)


#############################################################
# Define functions
#############################################################

def safe_float_to_int(val, fallback, scale=1.0):
    """
    Safely converts a value to an integer and applies a scaling factor. 
    If conversion fails, returns a fallback value.

    Parameters:
        val: The value to convert to an integer.
        fallback: The value to return if conversion is not possible.
        scale: A multiplier applied to the integer result. Default is 1.0.

    Returns:
        float: The converted and scaled value, or the fallback if conversion fails.
    """
    
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return fallback
        return int(val) * scale
    except (ValueError, TypeError):
        return fallback


def reset_simulation(state, both=False):
    """
    Resets the simulation state by clearing input fields and resetting the data table 
    for one or both scenarios.

    If `both` is True, both Scenario 1 and Scenario 2 are reset.
    If `both` is False, the scenario to reset is determined by `state.scenario_value`.
    If `state.scenario_value` is not recognized, Scenario 1 is reset by default.

    Parameters:
        state: An object containing the simulation state, including input fields and data tables for each scenario.
        both (bool, optional): If True, resets both scenarios. 
                               If False (default), resets the currently selected scenario based on `state.scenario_value`.

    Returns:
        None
    """
    
    # Create empty dataframe
    df = pd.DataFrame(columns=["Date", "Component", "Reorder Point", "Lead Time", "Safety Stock", "Demand", "Inventory", "Order Quantity"])

    def reset_scenario_1():
        state.input_inv1 = None
        state.input_ss1 = None
        state.input_lt1 = None
        state.override_order_qty1 = None
        state.override_order_qty_period1 = None
        state.table_data_scenario1 = df.copy()

    def reset_scenario_2():
        state.input_inv2 = None
        state.input_ss2 = None
        state.input_lt2 = None
        state.override_order_qty2 = None
        state.override_order_qty_period2 = None
        state.table_data_scenario2 = df.copy()

    if both:
        reset_scenario_1()
        reset_scenario_2()
        print("Reset both scenarios")
    else:
        if state.scenario_value == "Scenario 1":
            reset_scenario_1()
            print("Reset Scenario 1")
        elif state.scenario_value == "Scenario 2":
            reset_scenario_2()
            print("Reset Scenario 2")
        else:
            reset_scenario_1()
            print("Reset default (Scenario 1)")


def load_xlsx_file(state):
    """
    Loads data from an Excel file.
    If loading fails, an error message is printed.

    Parameters:
        state: An object containing the path to the Excel file (state.path) and a field to store the loaded data (state.new_data).

    Returns:
        None
    """

    try:
        state.new_data = pd.read_excel(state.path)
        print(state.new_data.head())
        run(state.new_data, input=True)
    except Exception as e:
        print(f"Fejl ved indlæsning af Excel: {e}")


def plot_machine(graph_data):
    """
    Plots machine demand over time using Plotly, distinguishing historical and predicted data.

    Parameters:
        graph_data: A pandas DataFrame where columns represent machine demand and the index is datetime-formatted.

    Returns:
        plotly.graph_objs._figure.Figure or None: A Plotly figure visualizing the demand if data is present; otherwise None.
    """

    if len(graph_data.columns) > 0: 
        figure = go.Figure()
        columns = graph_data.columns
        max_value = max([graph_data[col].values.max() for col in columns])
        forecast_year = graph_data.index.year.max()
        forecast_len = (graph_data.index.year == forecast_year).sum()
        split_index = len(graph_data) - forecast_len  # The index where the dashed line starts
        
        # Define colors
        plotly_colors = pc.qualitative.Plotly
        color_index = 0

        for col in columns:
            line_color = plotly_colors[color_index % len(plotly_colors)]
            
            # The historical values must be indicated by a solid line
            figure.add_trace(go.Scatter(
                x=graph_data.index[:split_index],
                y=graph_data[col].values[:split_index],
                name=f"{col}",
                mode='lines',
                line=dict(dash='solid', color=line_color),
                showlegend=True,
                hovertext=[f"Date: {x.strftime('%B %Y')},\nHistorical demand for {col}: {y:.2f}" 
                                for x, y in zip(graph_data.index[:split_index], graph_data[col].values[:split_index])],
                hoverinfo="text"))

            # The predicted values must be indicated by a dashed line
            figure.add_trace(go.Scatter(
                x=graph_data.index[split_index-1:],
                y=graph_data[col].values[split_index-1:],
                name=f"{col}",
                mode='lines',
                line=dict(dash='dash', color=line_color),
                showlegend=False,
                hovertext=[f"Date: {x.strftime('%B %Y')},\nPredicted demand for {col}: {y:.2f}" 
                                for x, y in zip(graph_data.index[split_index-1:], graph_data[col].values[split_index-1:])],
                hoverinfo="text"
            ))

            # Update color for the next iteration
            color_index += 1

        # Add dashed vertical line to separate historical values from predicted values
        figure.add_trace(go.Scatter(
            x=[graph_data.index[split_index-1], graph_data.index[split_index-1]],
            y=[0, max_value],
            mode='lines',
            line=dict(dash='dot', color='white'),
            showlegend=False
        ))  

        figure.update_layout(
            xaxis_title="Time",
            yaxis_title="Demand"
        )
    else:
        figure = None
    return figure


def plot_component(graph_data):
    """
    Plots component-wise demand over time using subplots, distinguishing between historical and predicted data.

    Parameters:
        graph_data: A pandas DataFrame where columns represent different components' demand, and the index is datetime-formatted.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure with individual subplots for each component's demand.
    """

    if len(graph_data.columns) > 0: 
        number_of_cols = 3
        number_of_plots = len(graph_data.columns)
        number_of_rows = math.ceil(number_of_plots / number_of_cols)
        forecast_year = graph_data.index.year.max()
        forecast_len = (graph_data.index.year == forecast_year).sum()
        split_index = len(graph_data) - forecast_len  # The index where the dashed line starts

        # Define colors
        plotly_colors = pc.qualitative.Plotly
        color_index = 0

        # Create the figure
        fig = make_subplots(
            rows=number_of_rows, cols=number_of_cols,
            subplot_titles=[str(char) for char in list(graph_data.columns)])
        
        # Define the subplots of the figure
        for i, component in enumerate(list(graph_data.columns), start=0):
            max_value = graph_data[component].values.max()

            row = i // number_of_cols + 1
            col_pos = i % number_of_cols + 1
                
            line_color = plotly_colors[color_index % len(plotly_colors)]

            # The historical values must be indicated by a solid line
            fig.add_trace(go.Scatter(
                x=graph_data.index[:split_index],
                y=graph_data[component].values[:split_index],
                name=f"Historical demand for {component}",
                mode='lines',
                line=dict(dash='solid', color=line_color),
                showlegend=False,
                hovertext=[f"Date: {x.strftime('%B %Y')}\nHistorical demand for {component}: {y:.2f}"
                                for x, y in zip(graph_data.index[:split_index], graph_data[component].values[:split_index])],
                hoverinfo="text"),
                row=row, 
                col=col_pos)

            # The predicted values must be indicated by a dashed line
            fig.add_trace(go.Scatter(
                x=graph_data.index[split_index-1:],
                y=graph_data[component].values[split_index-1:],
                name=f"Forecast: {component}",
                mode='lines',
                line=dict(dash='dash', color=line_color),
                showlegend=False,
                hovertext=[f"Date: {x.strftime('%B %Y')}\nPredicted demand for {component}: {y:.2f}"
                                for x, y in zip(graph_data.index[:split_index], graph_data[component].values[:split_index])],
                hoverinfo="text"),
                row=row, 
                col=col_pos)

            # Update color for the next iteration
            color_index += 1

            # Add dashed vertical line to separate historical values from predicted values
            fig.add_trace(go.Scatter(
                x=[graph_data.index[split_index-1], graph_data.index[split_index-1]],
                y=[0, max_value],
                mode='lines',
                line=dict(dash='dot', color='white'),
                showlegend=False),
                row=row, 
                col=col_pos)

        # Layout settings
        fig.update_layout(
            height=number_of_plots*100)
        
    return fig


def plot_inventory(graph_data):
    """
    Plots inventory levels and predicted demand over time using grouped bar charts.

    Parameters:
        graph_data: A dictionary where each key maps to another dictionary containing:
            - 'inventory': A pandas Series with datetime index and inventory values.
            - 'quantity': A pandas DataFrame with columns ['Order Month', 'Order Quantity'] and a datetime index.
            - 'demand' (optional): A pandas Series with datetime index and predicted demand values.

    Returns:
        plotly.graph_objs._figure.Figure or None: A Plotly bar chart visualizing inventory and demand, or None if input is empty or invalid.
    """

    if graph_data is None or len(graph_data) == 0:
        return None

    # Only use first key. Contains only one key, as only one component is visualized at a time.
    key = list(graph_data.keys())[0]
    data = graph_data[key]

    # Colors
    colors = pc.qualitative.Plotly

    fig = go.Figure()

    # Inventory bar
    fig.add_trace(go.Bar(
        x=data["inventory"].index,
        y=data["inventory"],
        name="Inventory",
        marker=dict(color=colors[0]),
        hovertext=[f"Date: {x.strftime('%B %Y')}\nInventory: {y:.2f}"
                        for x, y in zip(data["inventory"].index, data["inventory"])],
        hoverinfo="text"))

    # Demand bar (if available)
    if "demand" in data:
        fig.add_trace(go.Bar(
            x=data["demand"].index,
            y=data["demand"],
            name="Predicted Demand",
            marker=dict(color=colors[1]),
            hovertext=[f"Date: {x.strftime('%B %Y')}\nPredicted Demand: {y:.2f}"
                            for x, y in zip(data["demand"].index, data["demand"])],
            hoverinfo="text"))

    # Layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=245
    )

    fig.update_xaxes(tickformat="%b %Y")

    return fig