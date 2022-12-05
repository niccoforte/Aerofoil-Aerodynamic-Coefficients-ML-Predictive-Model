import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.interpolate as interp
import pandas as pd


class Profile:
    """
    Doc String
    """

    def __init__(self, file, points=51, x_val=None, prnt=False):
        self.file = file
        self.points = points
        self.x_val = None

        self.name = None
        self.coords_up = None
        self.coords_low = None
        self.x = None
        self.y = None

        self.spline_xs = None

        self.splines = None
        self.spline_funcs = None

        if prnt:
            print('Achieving profile coordinates...')
            self.coord_profile()
            print(f' {self.name} Done. Creating x-coordinate cosine distribution with {points} points...')
            self.x_distribution()
            print(f' Done. Interpolating upper and lower profile splines at coordinates...')
            self.get_spline()
            print(' Done.')
        else:
            self.coord_profile()
            self.x_distribution()
            self.get_spline()

    def coord_profile(self):
        f = open(self.file, "r")
        lines = f.readlines()
        dat = []
        empty_line_indxs = []
        indx = 0
        for line in lines:
            indx += 1
            line = line.strip()
            if line == '': empty_line_indxs.append(indx - 1)
            dat.append(line)

        if dat[0][-8:] == ' AIRFOIL':
            name = dat[0][0:-8]
        else:
            name = dat[0]

        #        if len(empty_line_indxs) == 0:
        #            f = open(self.file, 'w')
        #
        #
        #        else:
        f.close()

        coords_up = dat[empty_line_indxs[0] + 1:empty_line_indxs[1]]
        xs_up = [float(coord[:7]) for coord in coords_up]
        ys_up = [float(coord[-10:]) for coord in coords_up]
        sorter = pd.DataFrame({'x': xs_up, 'y': ys_up})
        sorter = sorter.sort_values('x')
        xs_up = list(sorter.x)
        ys_up = list(sorter.y)

        coords_low = dat[empty_line_indxs[1] + 1:]
        xs_low = [float(coord[:7]) for coord in coords_low]
        ys_low = [float(coord[-10:]) for coord in coords_low]
        sorter = pd.DataFrame({'x': xs_low, 'y': ys_low})
        sorter = sorter.sort_values('x')
        xs_low = list(sorter.x)
        ys_low = list(sorter.y)

        coords_up = [xs_up, ys_up]
        coords_low = [xs_low, ys_low]
        # print(coords_low[0])

        xs_low_rev = list(xs_low)
        xs_low_rev.reverse()
        x = xs_up + xs_low_rev
        ys_low_rev = list(ys_low)
        ys_low_rev.reverse()
        y = ys_up + ys_low_rev

        #        plt.figure(0)
        #        plt.plot(x,y)
        #        plt.ylim([min(y)-0.15, max(y)+0.15])
        #        plt.title(name)

        self.name = name
        self.coords_up = coords_up
        self.coords_low = coords_low
        self.x = x
        self.y = y
        return name, coords_up, coords_low, x, y

    def x_distribution(self):
        points = self.points

        spline_xs_lin = np.linspace(0.0, math.pi, points)
        spline_xs = 0.5 * (1 - np.cos(spline_xs_lin))

        self.spline_xs = spline_xs
        return spline_xs

    def get_spline(self):
        name = self.name
        coords_up = self.coords_up
        coords_low = self.coords_low
        x = self.x
        y = self.y
        spline_xs = self.spline_xs

        spline_up = interp.InterpolatedUnivariateSpline(coords_up[0], coords_up[1])
        yfunc_up = spline_up(spline_xs)
        spline_func_up = [spline_xs.round(5), yfunc_up.round(5)]

        spline_low = interp.InterpolatedUnivariateSpline(coords_low[0], coords_low[1])
        yfunc_low = spline_low(spline_xs)
        spline_func_low = [spline_xs.round(5), yfunc_low.round(5)]

        #        plt.figure(1)
        #        plt.title(name)
        #        # plt.plot(coords_up[0],coords_up[1])
        #        # plt.plot(coords_low[0],coords_low[1])
        #        # plt.scatter(x,y, s=1)
        #        plt.ylim([min(y)-0.15, max(y)+0.15])
        #        plt.grid()
        #        plt.plot(spline_xs, yfunc_up)
        #        plt.plot(spline_xs, yfunc_low)

        splines = [spline_up, spline_low]
        spline_funcs = [spline_func_up, spline_func_low]

        self.spline_funcs = spline_funcs
        self.splines = splines
        return splines, spline_funcs

# -------------------- Zoomed in inspection of tip and tail ------------------

#        plt.figure(2)
#        #plt.plot(x,y)
#        plt.title(name)
#        plt.scatter(x,y, s=10)
#        plt.ylim(-0.01, 0.01)
#        plt.xlim(0.95, 1.05)
#
#        plt.plot(xfunc_up, yfunc_up)
#        plt.plot(xfunc_low, yfunc_low)
#
#
#        plt.figure(3)
#        #plt.plot(x,y)
#        plt.title(name)
#        plt.scatter(x,y, s=10)
#        plt.ylim(-0.02, 0.02)
#        plt.xlim(-0.05, 0.05)
#
#        plt.plot(xfunc_up, yfunc_up)
#        plt.plot(xfunc_low, yfunc_low)

# ----------------------------------------------------------------------------


def splines_df():
    files = []
    for file in os.scandir('aerofoil_dat'):
        if file.name.endswith('.dat'):
            files.append(file.path)
    files = sorted(files)

    foil_names = []
    foil_xs = []
    foil_ys_up = []
    foil_ys_low = []
    foil_splines = []
    for file in files:
        try:
            p = Profile(file, points=51, prnt=False)

            name = p.name
            foil_names.append(name)

            spline_funcs = p.spline_funcs
            splinef_up = spline_funcs[0]
            splinef_low = spline_funcs[1]
            splines = p.splines

            foil_splines.append(splines)
            foil_xs.append(splinef_up[0])
            foil_ys_up.append(splinef_up[1])
            foil_ys_low.append(splinef_low[1])

        except:
            foil_names.append(file)
            foil_splines.append('Error')
            foil_xs.append('Error')
            foil_ys_up.append('Error')
            foil_ys_low.append('Error')

    aerofoil_profiles = {'name': foil_names, 'Xs': foil_xs, 'ys_up': foil_ys_up, 'ys_low': foil_ys_low,
                         'splines': foil_splines}
    aerofoil_profiles_df = pd.DataFrame(aerofoil_profiles)
    aerofoil_profiles_df = aerofoil_profiles_df

    return aerofoil_profiles_df


def plot_profile(df, indx, x_val=None):
    print(f'Plotting {df.name[indx]} Aerofoil...')

    plt.figure(1)
    plt.title(df.name[indx])
    plt.plot(df.Xs[indx], df.ys_low[indx])
    plt.plot(df.Xs[indx], df.ys_up[indx])
    plt.ylim([min(df.ys_low[indx]) - 0.15, max(df.ys_up[indx]) + 0.15])
    plt.grid()

    if x_val:
        print(f' and evaluating y coordinates at x={x_val}...')

        splines = df.splines[indx]
        spline_up = splines[0]
        spline_low = splines[1]

        y_val_up = spline_up(x_val)
        y_val_low = spline_low(x_val)
        print(f'  y_up = {y_val_up.round(5)} and y_low = {y_val_low.round(5)}.')

        plt.figure(1)
        plt.plot(x_val, y_val_up, marker='o', color='orangered')
        plt.plot(x_val, y_val_low, marker='o', color='orangered')
        plt.axvline(x=x_val, color='orangered')


aerofoils_df = splines_df()
# plot_profile(aerofoils_df, 4, x_val=0.38)
# aerofoils_df

worked = aerofoils_df[aerofoils_df['Xs'] != 'Error'].reset_index()
plot_profile(worked, 72, x_val=0.36)
# len(worked.name.tolist()), worked.name.tolist()