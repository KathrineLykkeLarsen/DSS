# DSS
A decision support tool for inventory management: Applying forecasting and inventory optimization

# Abstract
Implementing precise demand forecasting in SMEs can strengthen their competitiveness and growth potential. However, to effectively meet this demand, companies must also be able to optimize their inventory. The purpose of this paper is to examine how the combination of demand forecasting and inventory optimization can be used as a decision support tool in an SME. Four product groups are defined, and sales data for each is aggregated monthly and weekly - resulting in eight datasets. Lagged features are created and used as input for Gradient Boosting models. To ensure robust and generalizable models, hyperparameter tuning and cross-validation have been applied. Overall, seven of the eight models perform better than a naive baseline model, achieving a MASE score of less than 1. The predicted demand is used as input for a numerical simulation aimed at optimizing inventory and determining the order quantity that an employee should expect to place in order to meet the demand. Both components are visualized in a user interface, which includes a scenario analysis section. The results demonstrate that integrating demand forecasting with inventory optimization in a user interface provides SMEs with a practical decision support tool, helping them make informed decisions about inventory replenishment.

# Overview of Python files
DSS
- Initializes the DSS

BackendFunctionsGeneral
- Used for ML training, validation and prediction and inventory optimization

BackendFunctionsApp
- Helper functions for the DSS
