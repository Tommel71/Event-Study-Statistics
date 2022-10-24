import dataclasses
from functools import lru_cache
import pandas as pd
import numpy as np
import scipy.stats
import logging

# set logging to debug
logging.basicConfig(level=logging.DEBUG)


@dataclasses.dataclass
class TestResults:
    statistic: float
    pvalue: float


###################### adj-BMP TEST ############################

def calc_sigma_sq_AR_i(eps_i, M1_i):
    """
    :param eps_i: residuals of event i
    :param M1_i: non-missing values in eps_i
    :return: sigma_sq_AR_i
    """
    return (eps_i ** 2).sum() / (M1_i - 2)


def calc_SAR_i_t(AR_i_t, sigma_sq_AR_i, M1_i, R_market_estimation_window_centered_squared_sum,
                 R_market_event_day_centered_squared):
    """

    :param AR_i_t:  abnormal return of event i on day t
    :param sigma_sq_AR_i: variance of residuals of event i
    :param M1_i: non-missing values in eps_i
    :param R_market: market return
    :return: sigma_AR_i_t
    """
    return AR_i_t / (sigma_sq_AR_i * (1 + (1 / M1_i) + (
                R_market_event_day_centered_squared / (R_market_estimation_window_centered_squared_sum)))) ** (1 / 2)


def calc_z_BMP_t(SARs):
    """
    :param standardised_ARs: We calculated it only for the event day so we are not slicing for the day here anymore
    :return: the unadjusted z_BMP_E
    """
    N = len(SARs)
    ASAR_t = SARs.mean()
    return ASAR_t / (N**(1/2) * ((1 / (N - 1)) * ((SARs - ASAR_t) ** 2).sum()) ** (1 / 2))


def calculate_average_cross_correlation(eps, threshold=5_000_000):
    """

    :param eps: estimation window residuals
    :param threshold: Where to start subsampling, subsampling will pick threshold/2 datapoints at random without replacement and without the diagonal line
    :return: average cross correlation
    """

    @lru_cache(maxsize=None)  # to save half the work
    def cross_correlation(i, j):
        return np.correlate(eps[i, :], eps[j, :])

    eps = eps[np.random.permutation(eps.shape[0]), :]

    N = eps.shape[0]

    if N ** 2 < threshold:
        return np.mean([cross_correlation(i, j) for i in range(N) for j in range(i + 1, N) if i != j])

    print("Too many events to calculate average cross correlation, subsampling combinations...")
    # subsample tuples
    subset = set()
    len_subset = [0]

    def draw_tuple(N):
        # draw
        tuple_drawn = tuple(np.random.randint(0, N, 2))
        if tuple_drawn in subset:
            return

        if tuple_drawn[0] == tuple_drawn[1]:
            return

        subset.add(tuple_drawn)
        len_subset[0] += 1

    while len_subset[0] < threshold / 2:
        draw_tuple(N)

    subset = list(subset)

    return np.mean([cross_correlation(tuple_[0], tuple_[1]) for tuple_ in subset])


def adjust(z_BMP, eps):
    "Make adjustment to BMP test statistic"
    rho_bar_hat = calculate_average_cross_correlation(eps)
    N = eps.shape[0]
    adjusted = z_BMP * ((1 - rho_bar_hat) / (1 + (N - 1) * rho_bar_hat)) ** (1 / 2)
    return adjusted


# implement an adjusted böhmer test
def adjBMP_daily(AR, eps, R_market_estimation_window, R_market_event_window, t, adjustment=True):

    N = AR.shape[0]
    events = range(N)
    M1 = (~np.isnan(R_market_estimation_window)).sum(axis=1)
    num = (R_market_event_window[:, t] - R_market_estimation_window.mean(axis=1))**2
    denom = (((R_market_event_window.transpose() - R_market_estimation_window.mean(axis=1)).transpose())**2).sum(axis=1)
    AR_t = AR[:, t]
    s_sq_AR = [calc_sigma_sq_AR_i(eps[i], M1[i]) for i in events]
    S_AR_t = [(s_sq_AR[i]*(1+ 1/M1[i] + num[i]/denom[i]))**(1/2) for i in events]
    SAR_t = AR_t / S_AR_t
    ASAR_t = SAR_t.mean()
    S_ASAR_t = SAR_t.std(ddof=1)
    z_BMP_t = ASAR_t * (N**(1/2)/(S_ASAR_t))

    if adjustment:
        stat = adjust(z_BMP_t, eps)
    else:
        stat = z_BMP_t


    # find p-value for two-tailed test
    p = scipy.stats.norm.sf(abs(stat)) * 2  # two-tailed test, so we multiply by 2
    result = TestResults(stat, p)

    return result

def z_BMP_new(AR_, eps, R_market_estimation_window, R_market_event_window, CAR_period, adjustment=True):


    N = AR_.shape[0]
    AR = AR_.copy()[:, CAR_period[0]:(CAR_period[1] + 1)]
    M1 = (~np.isnan(R_market_estimation_window)).sum(axis=1)
    M2 = (~np.isnan(R_market_event_window)).sum(axis=1)

    CAR = AR.sum(axis=1)
    events = range(N)
    s_sq_AR = [calc_sigma_sq_AR_i(eps[i], M1[i]) for i in events]
    num = ((R_market_event_window.transpose() - R_market_estimation_window.mean()).transpose().sum(axis=1))**2
    denom = (((R_market_estimation_window.transpose() - R_market_estimation_window.mean(axis=1)).transpose())**2).sum(axis=1)
    S_CAR = np.asarray([(s_sq_AR[i]*(M2[i] + M2[i]**2/M1[i] + num[i]/denom[i]))**(1/2) for i in events])

    SCAR = CAR/S_CAR
    SCAR_bar = SCAR.mean()
    S_CAR_bar = SCAR.std(ddof=1)
    z_BMP = N**(1/2)*SCAR_bar/S_CAR_bar
    return TestResults(z_BMP, scipy.stats.norm.sf(abs(z_BMP))*2)

def adjBMP(AR_, eps, R_market_estimation_window, R_market_event_window, CAR_period, adjustment=True):
    # + because we let the user do [0,40] to get [0,40] and not [0,41], as this is python specific?
    # maybe bad idea to do this, but would have to check this in multiple places

    AR = AR_.copy()[:, CAR_period[0]:(CAR_period[1] + 1)]
    M1 = (~np.isnan(R_market_estimation_window)).sum(axis=1)
    M2 = (~np.isnan(R_market_event_window)).sum(axis=1)
    events = range(AR.shape[0])
    N = AR.shape[0]

    sigma_sq_AR = np.asarray([calc_sigma_sq_AR_i(eps[i], M1[i]) for i in events])

    CAR = AR.cumsum(axis=1)

    summand = (R_market_event_window - R_market_estimation_window.mean()).sum() ** 2 / (
            (R_market_estimation_window - R_market_estimation_window.mean()) ** 2).sum()


    S_CAR = [(sigma_sq_AR[i] * (M2[i] + (M2[i] ** 2 / M1[i]) + summand)) ** (1 / 2) for i in events]
    SCAR = (CAR.transpose() / S_CAR).transpose()

    # calculate the unadjusted z_BMP_E
    z_BMP = N ** (1 / 2) * SCAR.mean() / SCAR.std(ddof=1)


    if adjustment:
        stat = adjust(z_BMP, eps)
    else:
        stat = z_BMP


    # find p-value for two-tailed test
    p = scipy.stats.norm.sf(abs(stat)) * 2  # two-tailed test, so we multiply by 2
    result = TestResults(stat, p)

    return result


###################### GRANK TEST ############################


def calculate_SCAR_star(SCAR):
    "just standardize SCAR"
    return SCAR / SCAR.std(ddof=1)  # ddof=1 to get unbiased estimator of variance


def calculate_GSAR(SCAR, SAR, L1, t_1, tau):
    "repeat SCAR tau days and concatenate arrays"
    GSAR = np.concatenate([SAR[:, :L1 + t_1]] + [SCAR.reshape(-1, 1) for _ in range(tau)] + [SAR[:, L1 + t_1 + tau:]],
                          axis=1)
    return GSAR


def calculate_U(GSAR_T_script, M_T_script):
    "calculate ranks, divide by the number of valid days and then subtract 0.5. Should have mean 0."
    result = (scipy.stats.rankdata(GSAR_T_script, axis=1).transpose() / (M_T_script + 1)).transpose() - 0.5

    return result


def calculate_grank_Z(U, N, T_script):
    "calculate grank test statistic"
    U_bar = np.asarray([1 / N[t] * U[:, t].sum() for t in
                        T_script])  # TODO ASSUMING NO NANS FOR NOW, WITH NANS WE NEED LIKE A LIST AS U
    numerator = U_bar[-1]
    denominator = U_bar.std(ddof=0)  # use norming factor of 1/N and not 1/(N-1) here like in the paper
    return numerator / denominator


# implement generalised rank test
def grank(AR, eps, R_market_estimation_window, R_market_event_window, event_day, CAR_period,
          adjust_cumulating_ind_prediction_errors=False):
    """
    :param AR: abnormal returns
    :param eps: residuals from the market model
    :param R_market_estimation_window: returns of the market in the estimation window for each event
    :param R_market_event_window: returns of the market in the event window for each event
    :param event_day: day of the event, if day nr 21 is the event day, then event_day = 20
    :param CAR_period: period of the event window we are observing. Careful for 41 days [0, 40] would represent the full
    event window
    :param adjust_cumulating_ind_prediction_errors: Whether to use adjustment factor suggested by Mikkelson and
    Partch (1988), which makes no difference in this case, because we also calculate the SCAR_star, which is
    standardised by the standard deviation of the SCAR, therefore the scalar adjustment factor does not change the result.
    :return: test stats and p-value
    """
    M1 = (~np.isnan(R_market_estimation_window)).sum(axis=1)
    L1 = R_market_estimation_window.shape[1]  # TODO perhaps have to change this
    L2 = R_market_event_window.shape[1]

    R_market_event_day = R_market_event_window[:, event_day]
    events = range(R_market_estimation_window.shape[0])
    days = range(L1 + L2)

    sigma_sq_AR = np.asarray([calc_sigma_sq_AR_i(eps[i], M1[i]) for i in events])

    AR_estimation_and_event = np.concatenate([eps, AR], axis=1)

    R_market_bar = R_market_estimation_window.mean(axis=1)
    R_market_estimation_window_centered_squared_sum = (
                (R_market_estimation_window.transpose() - R_market_bar).transpose() ** 2).sum(axis=1)
    R_market_event_day_centered_squared = (R_market_event_day.transpose() - R_market_bar).transpose() ** 2

    calc_SAR = lambda i_t: calc_SAR_i_t(AR_estimation_and_event[i_t[0], i_t[1]], sigma_sq_AR[i_t[0]], M1[i_t[0]],
                                        R_market_estimation_window_centered_squared_sum[i_t[0]],
                                        R_market_event_day_centered_squared[i_t[0]])
    event_day_df = pd.DataFrame([[(i, t) for t in days] for i in events])

    SAR = event_day_df.applymap(calc_SAR).values

    tau = CAR_period[1] - CAR_period[0] + 1
    AR_period = AR[:, CAR_period[0]:(CAR_period[-1] + 1)]
    # CAR_period = AR_period.cumsum(axis=1)
    # CAR_tau = CAR_period[:,-1]
    CAR_tau = AR_period.sum(axis=1)

    if adjust_cumulating_ind_prediction_errors:
        adjustment_factor = (
                    L1 + L2 + L2 / L1 + ((R_market_event_window - R_market_estimation_window.mean()) ** 2).sum() / (
                    (R_market_estimation_window - R_market_estimation_window.mean()) ** 2).sum())

    else:
        adjustment_factor = 1  # no adjustment

    def calculate_SCAR(i):
        return CAR_tau[i] / (sigma_sq_AR[i] * adjustment_factor) ** (1 / 2)

    SCAR = np.asarray([calculate_SCAR(i) for i in events])
    SCAR_star = calculate_SCAR_star(SCAR)
    T_script = list(range(0, L1)) + [
        L1 + CAR_period[0]]  # for us T_script is the estimation window + the first day of the CAR period
    GSAR = calculate_GSAR(SCAR_star, SAR, L1, CAR_period[0], tau)

    M_T_script = (~np.isnan(GSAR[:, T_script])).sum(axis=1)
    N_T_script = (~np.isnan(GSAR[:, T_script])).sum(axis=0)  # just called N in the paper
    T_script_zero = range(len(T_script))
    U = calculate_U(GSAR[:, T_script], M_T_script)  # TODO ASSUMING NO NANS FOR NOW, WITH NANS WE NEED LIKE A LIST AS U

    grank_Z = calculate_grank_Z(U, N_T_script, T_script_zero)
    grank_t = grank_Z * ((L1 - 1) / (L1 - grank_Z ** 2)) ** (1 / 2)
    pvalue = scipy.stats.t.sf(abs(grank_t), L1 - 1) * 2  # two-tailed test, so we multiply by 2
    result = TestResults(grank_t, pvalue)

    return result
