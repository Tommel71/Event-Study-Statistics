import pickle
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from eventstudystatistics import adjBMP, adjBMP_daily, grank, z_BMP_new

def calculate_coefficients(X,Y):
    # implement ordinary least squares in numpy

    # add a constant to the X matrix
    X = np.c_[np.ones(X.shape[0]), X]

    # calculate the coefficients
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y

    # calculate the residuals
    eps = Y - X @ beta

    return beta[0], beta[1], eps



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

test_res2 = grank(AR, eps, estimation_window_market_return, event_window_market_return, event_day, CAR_period)
print(test_res2)



### SIMULATION:

### test if there are no missing values in the data, expecting high p value
adjBMP_results = []
grank_results = []

np.random.seed(3)
J = 5000

for j in range(J):
    print(j)
    event_window_market_return = np.random.normal(0, 0.1, (n_events, length_event_window))
    event_window_company_return = np.random.normal(0, 0.05, (n_events, length_event_window)) + event_window_market_return

    estimation_window_market_return = np.random.normal(0, 0.1, (n_events, length_estimation_window))
    estimation_window_company_return = np.random.normal(0, 0.05, (n_events, length_estimation_window)) + estimation_window_market_return

    AR_ = []
    eps_ = []
    print("calculate AR...")
    for i in range(n_events):
        alpha, beta, eps = calculate_coefficients(estimation_window_market_return[i, :],
                                                      estimation_window_company_return[i, :])
        ## Calculate the abnormal returns
        abnormal_return = event_window_company_return[i, :] - alpha - beta * event_window_market_return[i, :]
        AR_.append(abnormal_return)
        eps_.append(eps)

    print("Done calculating AR")
    AR = np.asarray(AR_)
    eps = np.asarray(eps_)

    test_res = z_BMP_new(AR, eps, estimation_window_market_return, event_window_market_return, CAR_period, adjustment=False)
    adjBMP_results.append(test_res)

    #test_res_daily = adjBMP_daily(AR, eps, estimation_window_market_return, event_window_market_return, event_day, adjustment=False)
    #adjBMP_results.append(test_res_daily)

    #test_res2 = grank(AR, eps, estimation_window_market_return, event_window_market_return, event_day, CAR_period)
    #grank_results.append(test_res2)



adj_BMP_z_stat = np.asarray([res.statistic for res in adjBMP_results])
grank_t_stat = np.asarray([res.statistic for res in grank_results])

# histogram of the statistics
plt.hist(adj_BMP_z_stat, bins=200, density=True)
# add a standard normal distribution
x = np.linspace(-5, 5, length_estimation_window)
plt.plot(x, norm.pdf(x))

plt.show()

plt.hist(grank_t_stat, bins=200, density=True)
# add a student t distribution of 99 degrees of freedom
x = np.linspace(-5, 5, length_estimation_window)
plt.plot(x, t.pdf(x, length_estimation_window-1))

plt.show()
