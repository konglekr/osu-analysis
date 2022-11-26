from ossapi import *
from scipy import optimize
from scipy import interpolate
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import io
import base64

matplotlib.use('Agg')
plt.style.use("dark_background")
matplotlib.rc('font', serif='Candara') 

api = OssapiV2(19003, 'TVMi13ZOzWLvCo1CUMTg4Bs49r0ytI1PZ0INDr0d')

params = np.array([6.27996856e+02, 3.56978771e-01])

with open('pp.pickle', 'rb') as pp_file:
    pp = pickle.load(pp_file)

with open('time.pickle', 'rb') as time_file:
    time = pickle.load(time_file)

center_pp = pp[(time>np.percentile(time, 10)) & (time<np.percentile(time, 90))]
center_time = time[(time>np.percentile(time, 10)) & (time<np.percentile(time, 90))]

def corr(x, change): 
    return (params[0] + change) * x ** params[1]

def rev_corr(x, pp):
    return (pp * x ** (-params[1])) - params[0]

def find_percentile(change, desired):
    within = 0
    for i in range(len(center_time)):
        if center_pp[i] <= corr(center_time[i], change):
            within += 1
    
    return abs(within/len(center_time) - desired)

def find_reverse_percentile(desired, change):
    return find_percentile(change, desired)

def plot_user_stats(all_pp, all_time, user_pp, user_time, my_pct, user_name):
    img1 = io.BytesIO()

    plt.clf()
    plt.scatter(all_time, all_pp, s = .1, c = 'cornflowerblue', label = "Other players")
    plt.scatter(user_time, user_pp, s = 10, c = 'red', label = 'You')
    plt.title(f"{user_name}'s PPP: {my_pct * 100:.2f}%")
    plt.xlabel("Playing Time (Hr)")
    plt.ylabel("Performance Points")
    plt.savefig(img1, format='png')
    img1.seek(0)
    
    img1 = base64.b64encode(img1.getvalue()).decode()

    img2 = io.BytesIO()

    plt.clf()
    x = np.arange(0, 20000, 1000)
    y = [optimize.minimize_scalar(find_reverse_percentile, args = (rev_corr(user_time, i)), tol = .00001).x for i in x]
    X_Y_Spline = interpolate.make_interp_spline(x, y)
    x_plot = np.arange(0, 20000)
    y_plot = X_Y_Spline(x_plot)
    plt.plot(x_plot, y_plot)
    plt.scatter(user_pp, my_pct)
    plt.title(f"PPP Distribution at time: {user_time:.2f} hrs")
    plt.xlabel("Performance Points")
    plt.ylabel("Percentile")
    plt.savefig(img2, format='png')
    img2.seek(0)
    
    img2 = base64.b64encode(img2.getvalue()).decode()

    return "data:image/png;base64,{}".format(img1), "data:image/png;base64,{}".format(img2)


def stats(user):
    try:
        user_data = api.user(user).statistics
    except:
        user = 'peppy'
        user_data = api.user(user).statistics
    change = rev_corr(user_data.play_time/3600, user_data.pp)
    my_pct = optimize.minimize_scalar(find_reverse_percentile, args = (change), tol = .00001)
    return plot_user_stats(pp, time, user_data.pp, user_data.play_time / 3600, my_pct.x, user)