import pickle
import numpy as np
from eventstudystatistics import adjBMP, adjBMP_daily, grank

def calculate_coefficients(X,Y):
    # implement ordinary least squares in numpy

    # add a constant to the X matrix
    X = np.c_[np.ones(X.shape[0]), X]

    # calculate the coefficients
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y

    # calculate the residuals
    eps = Y - X @ beta

    return beta[0], beta[1], eps

### Generating random data for a showcase how to use the functions

CAR_period = [0, 40]  # including both edges, CAREFUL, this is not like python indexing, this is including the right side.
n_events = 120
length_event_window = 41 # L2
length_estimation_window = 100 # L1
event_day = 20 # within the event window index 20 is the event day

# single example for all tests:

event_window_market_return = np.random.normal(0, 0.1, (n_events, length_event_window))
event_window_company_return = np.random.normal(0, 0.05, (n_events, length_event_window)) + event_window_market_return

estimation_window_market_return = np.random.normal(0, 0.1, (n_events, length_estimation_window))
estimation_window_company_return = np.random.normal(0, 0.05, (n_events, length_estimation_window)) + estimation_window_market_return

AR_ = []
eps_ = []
print("calculate abnormal returns...")
for i in range(n_events):
    alpha, beta, eps = calculate_coefficients(estimation_window_market_return[i, :],
                                                  estimation_window_company_return[i, :])
    ## Calculate the abnormal returns
    abnormal_return = event_window_company_return[i, :] - alpha - beta * event_window_market_return[i, :]
    AR_.append(abnormal_return)
    eps_.append(eps)

print("Done calculating abnormal returns")
AR = np.asarray(AR_)
eps = np.asarray(eps_)


test_res = adjBMP_daily(AR, eps, estimation_window_market_return, event_window_market_return, event_day)
print(test_res)

test_res = adjBMP(AR, eps, estimation_window_market_return, event_window_market_return, CAR_period)
print(test_res)

test_res2 = grank(AR, eps, estimation_window_market_return, event_window_market_return, CAR_period)
print(test_res2)


### More realistic looking data:

# load pickle file from tests, feel free to inspect the params variable to see what the input looks like
with open("tests/params_adjbmp_grank.pkl", "rb") as f:
    params = pickle.load(f)

test_realistic = grank(*params)
test_realistic2 = adjBMP(*params)
