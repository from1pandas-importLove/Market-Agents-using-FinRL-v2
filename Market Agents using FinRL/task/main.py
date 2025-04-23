# import librabries and set the device
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE
import optuna

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

# preprocess data
# You can use the follow implementation for the stage 1
# or use the one you implemented on stage 1

# ------------------------------------------------------------------------------------------------

# Define your custom stock list
# EX: "AAPL", "MSFT", "GOOGL", "AMZN"
# Choose the ones, that you want to work with
CUSTOM_STOCK_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB",
    "TSLA", "BRK.B", "JNJ", "V", "WMT"
]


# Download stock data
START_DATE = '2011-01-01' # start date of the portfolio
END_DATE = '2022-12-31' # end  date of the portfolio
portfolio_raw_df = YahooDownloader(start_date=START_DATE,
                                   end_date=END_DATE,
                                   ticker_list=CUSTOM_STOCK_LIST).fetch_data()

# Group by ticker and count occurrences
portfolio_raw_df.groupby("tic").count()

# Normalize the data
# You can use GroupByScaler with a MaxAbsScaler here
portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(portfolio_raw_df)

# Select relevant columns
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]

# Split data into training and testing sets

START_DATE_TRAIN = "2011-01-01" # you start date for the train data
END_DATE_TRAIN = "2019-12-31" # your end date for the train data
START_DATE_TEST = "2020-01-01" # your start date for the test data
END_DATE_TEST = "2022-12-31" # your end date for the test data

df_portfolio_train = df_portfolio[(df_portfolio["date"] >= START_DATE_TRAIN) & (df_portfolio["date"] < END_DATE_TRAIN)]
df_portfolio_test = df_portfolio[(df_portfolio["date"] >= START_DATE_TEST) & (df_portfolio["date"] < END_DATE_TEST)]

# train the model
# Define the environment
# We will use portfolio optimization for the project
INITIAL_AMOUT = 100000 # initial amount of money in the portfolio: float
COMISSION_FEE_PTC = 0.001 # commission fee: float
TIME_WINDOW = 50# time window: int
FEATURES = ["close", "high", "low"] # ex: "close", "high"

environment_train = PortfolioOptimizationEnv(
    df_portfolio_train,
    initial_amount=INITIAL_AMOUT,
    comission_fee_pct=COMISSION_FEE_PTC,
    time_window=TIME_WINDOW,
    features=FEATURES,
    normalize_df=None # df is already normalized
)

# Set PolicyGradient parameters
# Set the learning rate for the training
model_kwargs = {
    "lr": 0.01, # put a learning rate, ex: 0.01
    "policy": EIIE, # we will use EIIE policy for this project
}

# Set EIIE's parameters
policy_kwargs = {
    "k_size": 3, # put the k_size: int
    "time_window": TIME_WINDOW, # time window defined previously
}

# Instantiate the model
model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)

# Train the model
EPISODES = 10 # number of episodes to training the model: in
DRLAgent.train_model(model, episodes=EPISODES)

# Save the model
torch.save(model.train_policy.state_dict(), "policy_EIIE.pt")

# evaluate the model
# Evaluate the model
# Create test env

INITIAL_AMOUT = 100000 # initial amount of money in the portfolio: float
COMISSION_FEE_PTC = 0.001 # commission fee: float
TIME_WINDOW = 50# time window: int
FEATURES = ["close", "high", "low"] # ex: "close", "high"

enviroment_test = PortfolioOptimizationEnv(
    df_portfolio_test,
    initial_amount=INITIAL_AMOUT,
    comission_fee_pct=COMISSION_FEE_PTC,
    time_window=TIME_WINDOW,
    features=FEATURES,
    normalize_df=None # df is already normalized
)


EIIE_results = {
    "train": environment_train._asset_memory["final"],
    "test": {},
}

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
policy = EIIE(time_window=TIME_WINDOW, device=device)
policy.load_state_dict(torch.load("policy_EIIE.pt"))

# testing
DRLAgent.DRL_validation(model, enviroment_test, policy=policy)
EIIE_results["test"] = enviroment_test._asset_memory["final"]

# print the final value of the portfolio
final_portfolio_value_test = enviroment_test._asset_memory["final"][-1]
print("The final portfolio value at test is:", final_portfolio_value_test)

# Hyperparameter tuning with optuna
EPISODES = 10# number of episodes to train the model

# Define the objective function
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2) # use trial to suggest values for lr
    k_size = trial.suggest_categorical("k_size", [3, 5, 7]) # use trial to suggest values for k_size
    time_window = trial.suggest_int("time_window", 30, 60, step=10) # use trial to suggest values for time_window

    # set up train environment
    # Training environment
    environment_train = PortfolioOptimizationEnv(
    df_portfolio_train,
    initial_amount=INITIAL_AMOUT,
    comission_fee_pct=COMISSION_FEE_PTC,
    time_window=time_window,
    features=FEATURES,
    normalize_df=None
    )

    # setup model kwargs
    model_kwargs = {
        "lr": lr,
        "policy": EIIE,
    }
    # setup policy kwargs
    policy_kwargs = {
        "k_size": k_size,
        "time_window": time_window,
    }
    # Train model using DRLAgent
    model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
    DRLAgent.train_model(model, episodes=EPISODES)

    # define test environment
    environment_test = PortfolioOptimizationEnv(
    df_portfolio_test,
    initial_amount=INITIAL_AMOUT,
    comission_fee_pct=COMISSION_FEE_PTC,
    time_window=time_window,
    features=FEATURES,
    normalize_df=None)

    # validate with test environment
    DRLAgent.DRL_validation(model, environment_test)

    # final portfolio value as metric for the optimization
    return environment_test._asset_memory["final"][-1]


# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
N_TRIALS = 20# select the number of trails for the optimization
study.optimize(objective, n_trials=N_TRIALS)

# print the best hyperparameters
BEST_HYPERPARAMETERS = {'lr': 0.009989562388670565, 'k_size': 5, 'time_window': 50} # using the study get the best hyperparameters
print("Best hyperparameters: ", BEST_HYPERPARAMETERS)
