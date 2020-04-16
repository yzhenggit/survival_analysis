import numpy as np

def unique_sorted_detection(data1, delta1, data2, delta2):
    """
    Find the unique detection values in the combined sample of {data1, data2},
    which will be used to calculate the linear rank statistic, see Table 3(A)
    in Feigelson & Nelson (1985).

    History:
    04/16/2020, Yong Zheng, UCB.
    """
    data1_detection = data1[delta1 == 1]
    data2_detection = data2[delta2 == 1]
    uni_y = np.unique(np.concatenate([data1_detection, data2_detection]))
    uni_y = np.sort(uni_y)
    return uni_y

def calc_n1j_n2j_nj(data1, delta1, data2, delta2):
    """
    Calculate n1j, n2j, and nj using Eq. 17 in Feigelson & Nelson (1985).

    Input:
    data1: dataset 1, right-censored (transformed from left-censored data)
    delta1: the detection array for dataset 1, 1 for detection,
            0 for lowerlimit (if right-censored) or upper limit (if left-censored)

    Output:
    n1j, n2j, nj

    History:
    04/16/2020, Yong Zheng, UCB.
    """

    uni_y = unique_sorted_detection(data1, delta1, data2, delta2)

    # calculate n1j using data1, it means the number of points in data1 with values higher
    # equal to xk in the uni_y arrays.
    n1j = np.zeros(uni_y.size, dtype='int')
    n2j = np.zeros(uni_y.size, dtype='int')
    for i in range(uni_y.size):
        n1j[i] = data1[data1>= uni_y[i]].size
        n2j[i] = data2[data2>= uni_y[i]].size

    nj = n1j + n2j
    return n1j, n2j, nj

def calc_m1j_m2j_mj(data1, delta1, data2, delta2):
    """
    Calculate m1j, m2j, and mj using Eq. 17 in Feigelson & Nelson (1985).
    m1j is the number of censored observations from data 1 which are between y_j and y_j+1

    Input:
    data1: the input dataset 1, right-censored (transformed from left-censored data)
    delta1: the detection array for dataset 1, 1 for detection,
            0 for lowerlimit (if right-censored) or upper limit (if left-censored)

    Output:
    m1j, m2j, mj

    History:
    04/16/2020, Yong Zheng, UCB.

    """

    uni_y = unique_sorted_detection(data1, delta1, data2, delta2)

    # calculate m1j using data1, it means the number of censored points in data1 with
    # values between x_k and x_k+1 in the uni_y arrays
    m1j = np.zeros(uni_y.size, dtype='int')
    data1_censored = data1[delta1 == 0]

    m2j = np.zeros(uni_y.size, dtype='int')
    data2_censored = data2[delta2 == 0]
    for i in range(uni_y.size):
        if i == uni_y.size - 1:
            # if it's the last data point, m1j is always 0 according to the definition
            m1j[i] = 0
            m2j[i] = 0
        else:
            yi = uni_y[i]
            yi_next = uni_y[i+1]

            # data 1
            # here, Eq. 17 actally says, yi < xik < y_j+1. However, in the case of
            # ties exist between censored and uncensored data (in this case, we have
            # one value of >6, and the other of 6), it's necessary to use
            # >= instead of > to count the >6 one. See calc_d1j_d2j_dj, and the line
            # above Eq. 8 in FN95.
            ind1 = np.all([data1_censored>=yi, data1_censored<yi_next], axis=0)
            if len(ind1) == 0:
                m1j[i] = 0
            else:
                m1j[i] = data1_censored[ind1].size

            # data 2
            ind2 = np.all([data2_censored>=yi, data2_censored<yi_next], axis=0)
            if len(ind2) == 0:
                m2j[i] = 0
            else:
                m2j[i] = data2_censored[ind2].size

    mj = m1j+m2j

    return m1j, m2j, mj

def calc_d1j_d2j_dj(data1, delta1, data2, delta2):
    """
    Calculate d1j, d2j, and dj using Eq. 17 in Feigelson & Nelson (1985).

    Input:
    data1: the input dataset 1, right-censored (transformed from left-censored data)
    delta1: the detection array for dataset 1, 1 for detection,
            0 for lowerlimit (if right-censored) or upper limit (if left-censored)

    Output:
    d1j, d2j, dj

    History:
    04/16/2020, Yong Zheng, UCB.

    """

    uni_y = unique_sorted_detection(data1, delta1, data2, delta2)

    # calculate d1j using data1, d1j means the number of points in data1 with values equal
    # to xk in the unique_x arrays.
    d1j = np.zeros(uni_y.size, dtype='int')
    d2j = np.zeros(uni_y.size, dtype='int')
    for i in range(uni_y.size):
        # so here, the 2nd condition of delta1==1 is not listed in Eq. 17, but according
        # to the line above Eq. 8, "ties between censored and uncensored values are broken
        # by considering the censored values to be larger", which means, if there are censored
        # and uncensored data points with the same values, here the censored points will not
        # be counted because we consider them larger.
        d1j[i] = data1[(data1==uni_y[i]) & (delta1==1)].size
        d2j[i] = data2[(data2==uni_y[i]) & (delta2==1)].size


    dj = d1j + d2j
    return d1j, d2j, dj

def calc_wj_Gehan(data1, delta1, data2, delta2):
    """
    Gehan weights defined by Eq 19 in Feigelson & Nelson (1985).

    History:
    04/16/2020, Yong Zheng, UCB.
    """

    n1j, n2j, nj = calc_n1j_n2j_nj(data1, delta1, data2, delta2)
    wj = nj
    return wj

def calc_wj_Logrank(data1, delta1, data2, delta2):
    """
    Logrank weights defined by Eq 19 in Feigelson & Nelson (1985).

    History:
    04/16/2020, Yong Zheng, UCB.
    """

    uni_y = unique_sorted_detection(data1, delta1, data2, delta2)
    wj = np.ones(uni_y.size, dtype='int')
    return wj


def calc_Ln(data1, delta1, data2, delta2, weight='Gehan'):
    """
    Gehan or Logrank statistics defined in page 3 of FN85.
    If Gehan test, wj should be calculated use calc_wj

    History:
    04/16/2020, Yong Zheng, UCB.
    """
    d1j, d2j, dj = calc_d1j_d2j_dj(data1, delta1, data2, delta2)
    n1j, n2j, nj = calc_n1j_n2j_nj(data1, delta1, data2, delta2)
    if weight.lower() == 'gehan':
        wj = calc_wj_Gehan(data1, delta1, data2, delta2)
    elif weight.lower() == 'logrank':
        wj = calc_wj_Logrank(data1, delta1, data2, delta2)
    else:
        print("weight has to be set to Gehan or Logrank.")
        sys.exit()

    Ln_j = wj*(d1j-dj*n1j/nj)
    Ln = np.nansum(Ln_j)
    return Ln_j, Ln

def calc_sigman_sq(data1, delta1, data2, delta2, weight='Gehan'):
    """
    variances of Gehan or Logrank statistics, as defined by Eq 20 in FN85.

    History:
    04/16/2020, Yong Zheng, UCB.
    """
    d1j, d2j, dj = calc_d1j_d2j_dj(data1, delta1, data2, delta2)
    n1j, n2j, nj = calc_n1j_n2j_nj(data1, delta1, data2, delta2)
    if weight.lower() == 'gehan':
        wj = calc_wj_Gehan(data1, delta1, data2, delta2)
    elif weight.lower() == 'logrank':
        wj = calc_wj_Logrank(data1, delta1, data2, delta2)
    else:
        print("weight has to be set to Gehan or Logrank.")
        sys.exit()

    sigman_sq_j = dj*(wj**2)*(n1j/nj)*(n2j/nj)*(nj-dj)/(nj-1)
    sigman_sq = np.nansum(sigman_sq_j)

    return sigman_sq_j, sigman_sq

def calc_p_value(Ln, sigman, weight='Gehan'):
    """
    Calculate the p value for the two-sample Gehan or Loghank tests, based on FN85.

    History:
    04/16/2020, Yong Zheng, UCB.
    """
    # first construct a normal distribution curve, also called bell curve
    from scipy import stats
    import numpy as np
    z_val = np.mgrid[-100:100:0.1]
    dz = np.mean(z_val[1:]-z_val[:-1])
    normal_pdf = stats.norm.pdf(z_val)
    # area = np.sum(normal_pdf*dz)
    # print("area below curve (has to be 1): ", area)

    # evaulate the significant to reject null hypothesis
    z_half_alpha = np.abs(Ln/sigman)
    area = np.sum(normal_pdf[np.all([z_val>=-z_half_alpha, z_val<=z_half_alpha], axis=0)]*dz)
    alpha = 1-area
    p = alpha
    print('%s test finds p=%.10f to reject H0\n'%(weight, p))
    return p

def two_sample_test_censored_data(data1, delta1, data2, delta2, weight='Gehan', censored_type='upper_limit'):
    """
    Use Gehan or Logrank two-sample tests to calculate the p value to test the null hypothesis
    that the input two data samples are drawn from the same (unknown) distribution.
    This is a nonparametric univariate survival analysis based on Keplan-Meier product-limit (PL) estimtor.

    Note that, if your datasets are too small (N<=10) or heavily censored (>90% of the data are limits),
    then this test yields unreliable result.

    Input:
    data1 and data2 can included censored data. They can be left-censored (upper limits) or
    right-censored (lower limits), but can't include both.
    censored_type: if your data include upper limits, then censored_type=upper_limit;
                   if your data include lower limits, then censored_type=lower_limit.
                   default to upper_limit, because most of the astro datasets are upper limit (non detection)

    History:
    04/16/2020, Yong Zheng, UCB.
    """

    data1 = np.array(data1)
    delta1 = np.array(delta1)
    data2 = np.array(data2)
    delta2 = np.array(delta2)

    if censored_type == 'upper_limit': # left censored
        # find the maximum values in the two survey
        max_val = np.max(np.concatenate([data1, data2]))

        # now make the data right censored
        data1 = max_val - data1
        data2 = max_val - data2

    # get the Gehan or the Logrank statistics
    Ln_j, Ln = calc_Ln(data1, delta1, data2, delta2, weight=weight)
    sigman_sq_j, sigman_sq = calc_sigman_sq(data1, delta1, data2, delta2, weight=weight)
    sigman = np.sqrt(sigman_sq)
    print('%s: Ln=%.4f+/-%.4f'%(weight, Ln, sigman))

    # calculate the p value
    p_val = calc_p_value(Ln, sigman, weight=weight)

    # let's give it some warning
    if (data1.size <= 10) or (data2.size <= 10):
        print(">> Warnin: one or both of the datasets have sample size <=10, this test maybe unreliable.")

    frac1_censored = data1[delta1==0].size/data1.size
    frac2_censored = data2[delta2==0].size/data2.size
    if (frac1_censored>0.9) or (frac2_censored>0.9):
        print(">> Warnin: one or both of the datasets are heavily censored (>90% of values are limits), this test maybe unreliable.")
    return p_val
