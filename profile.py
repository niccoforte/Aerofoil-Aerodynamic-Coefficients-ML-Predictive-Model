import numpy as np
import math
import scipy.interpolate as interp
import pandas as pd
import os


def create_profiles(directory='aerofoil_dat', ext='dat', points=51, prnt=False):
    """TODO: implement"""

    profiles = []
    for file in os.scandir(directory):
        if file.name.endswith('.' + ext):
            profiles.append(Profile(file, points=points, prnt=prnt))

    return profiles


class Profile:
    """ TODO: add docstring"""

    # Static variables
    dataframe = pd.DataFrame(columns=['name', 'x', 'y_high', 'y_low', 'spline'])

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
            try:
                print('Achieving profile coordinates...')
                self.coord_profile()
                print(f' {self.name} Done. Creating x-coordinate cosine distribution with {points} points...')
                self.x_distribution()
                print(f' Done. Interpolating upper and lower profile splines at coordinates...')
                self.get_spline()
                print('Updating static dataframe...')
                self.update_dataframe()
                print(' Done.')
            except:
                new_row = pd.DataFrame({'name': [self.file], 'x': 'Error', 'y_high': 'Error', 'y_low': 'Error',
                                        'spline': 'Error'})
                Profile.dataframe = pd.concat([Profile.dataframe, new_row])
                return
        else:
            try:
                self.coord_profile()
                self.x_distribution()
                self.get_spline()
                self.update_dataframe()
            except:
                new_row = pd.DataFrame({'name': [self.file], 'x': 'Error', 'y_high': 'Error', 'y_low': 'Error',
                                        'spline': 'Error'})
                Profile.dataframe = pd.concat([Profile.dataframe, new_row])
                return

    def update_dataframe(self):
        splinef_up = self.spline_funcs[0]
        splinef_low = self.spline_funcs[1]

        new_row = pd.DataFrame({'name': [self.name], 'x': [splinef_up[0]], 'y_high': [splinef_up[1]],
                                'y_low': [splinef_low[1]], 'spline': [self.splines]})
        Profile.dataframe = pd.concat([Profile.dataframe, new_row])

    def coord_profile(self):
        with open(self.file, "r") as f:
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