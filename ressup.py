import numpy as np
import pandas as pd
from ressuputils import *
import matplotlib.pyplot as plt


def normalize(arr):
    'arr - numpy.array'
    mu = np.mean(arr)
    st = np.std(arr)
    return ((arr - mu) / st), mu, st

def denorm(arr, mu, st):
    return (arr * st) + mu


# def slopeRes(xs, ys, m, b, loss):
#     'returns m, b for the line of residual adjusted best fit'

#     return map(float, model.layers[0].get_weights())

def ressup(DF, period, slopes=False):
    '''
    Returns the Support and Resistance lines for a given price history

            Parameters:
                    DF (pandas.DataFrame): Price History
                    period (int): number of last points for which to contruct the lines

            Returns:
                    srlDF (pandas.DataFrame): DataFrame with Support and Resistance values

            if Slopes == True Returns: 
                    srlDF, (supm, supb), (resm, resb): DF and line parameters (m, b) from `y = m * x + b`
    '''
    try:
        df = pd.DataFrame(DF.tail(period)['Close'])
    except KeyError:
        df = pd.DataFrame(DF.tail(period)['close'])

    x, mx, sx = normalize(df.index)
    try:
        y, my, sy = normalize(df['Close'].values)
    except KeyError: 
        y, my, sy = normalize(df['close'].values)
    # PRODUCE m, b for Sup Res Lines
    points = list(zip(x, y))

    lh_points = lower_hull(points)
    uh_points = upper_hull(points)

    lh_lines = [slope_intercept(lh_points[i], lh_points[i+1]) for i in range(0,len(lh_points)-1)]
    uh_lines = [slope_intercept(uh_points[i], uh_points[i+1]) for i in range(0,len(uh_points)-1)]

    closest_lh_line = min(lh_lines, key=lambda x: mse(x, points)) # The line closest to center of mass (at origin since normalized)
    closest_uh_line = min(uh_lines, key=lambda x: mse(x, points))

    XS = np.linspace(min(x), max(x))
    #for line in lh_lines:
    #    plt.plot(XS, line[0] * XS + line[1], color="blue")

    df['Support'] = ((closest_lh_line[0] * x + closest_lh_line[1])) * sy + my
    df['Resistance'] = ((closest_uh_line[0] * x + closest_uh_line[1])) * sy + my


    # plt.plot(x,y)
    # plt.scatter(*zip(*lh_points), marker="s", color="red")
    # plt.scatter(*zip(*uh_points), marker="x", color="purple")
    # plt.plot(XS, closest_lh_line[0] * XS + closest_lh_line[1], color="orange", label='NewSupport')
    # plt.plot(XS, closest_uh_line[0] * XS + closest_uh_line[1], color="green", label='NewResistance')

    # plt.tight_layout()
    # plt.legend()
    # plt.show()


    if slopes:
        #(supm, supb) = (closest_lh_line[0] * sy, closest_lh_line[1] * sy + my)
        (supm, supb) = (closest_lh_line[0]*sy/sx, -closest_lh_line[0]*sy/sx * mx + closest_lh_line[1] * sy + my)
        #(resm, resb) = (closest_uh_line[0] * sy, closest_uh_line[1] * sy + my)
        (resm, resb) = (closest_uh_line[0]*sy/sx, -closest_uh_line[0]*sy/sx * mx + closest_uh_line[1] * sy + my)
        return df, (supm, supb), (resm, resb)
    else:
        return df



if __name__ == '__main__':
    import random
    #data = [(0, 0), (1, 5), (2, 4), (3, 6), (4, 2), (5,3)]
    datalen = 100
    data = [(0, 0)]
    for i in range(1, 100):
        data.append((i, data[-1][1] + random.gauss(0, 3)))
    #data = [(i, random.randint(-datalen, datalen)) for i in range(datalen)]

    xs = [each[0] for each in data]
    ys = [each[1] for each in data]
    DF = pd.DataFrame({'Close': ys}, dtype=np.float64, index=xs)
    srlDF = srl(DF, datalen)

    #print(srlDF.head())

    srlDF.plot()
    print(DF.head())
    plt.title("SRL")

    #plt.tight_layout()
    #plt.legend()
    plt.show()

