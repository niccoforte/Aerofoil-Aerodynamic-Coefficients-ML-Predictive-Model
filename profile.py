import numpy as np
import math
import scipy.interpolate as interp
from matplotlib import pyplot as plt
import pandas as pd
import os


def create_profiles(directory='aerofoil_dat', ext='dat', points=51, prnt=False):
    """Generates a list of profile objects from a directory containing given files.

    Parameters
    ----------
    directory : str
        Directory in the project workspace that contains the files to be upoaded.
    ext : str
        Extension of the files.
    points : int
        Points at which to evaluate the splines of the aerofoils.
    prnt : bool
        Print log as profile objects are created.

    Returns
    -------
    list of Profile, pd.DataFrame
    """

    profiles = []
    for file in os.scandir(directory):
        if file.name.endswith('.' + ext):
            profiles.append(Profile(file, points=points, prnt=prnt))

    return profiles, Profile.dataframe


def plot_profile(df, indx, x_val=None):
    """Plots the profile of a given aerofoil.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the aerofoils' data.
    indx : int
        Index of the aerofoil to be plotted.
    x_val : float, optional
        x-value to be plotted along with the aerofoil.
    """

    print(f'Plotting {df.name[indx]} Aerofoil...')

    plt.figure(1)
    plt.title(df.name[indx])
    plt.plot(df.x[indx], df.y_low[indx])
    plt.plot(df.x[indx], df.y_up[indx])
    plt.ylim([min(df.y_low[indx]) - 0.15, max(df.y_up[indx]) + 0.15])
    plt.grid()

    if x_val:
        print(f' and evaluating y coordinates at x={x_val}...')

        splines = df.spline[indx]
        spline_up = splines[0]
        spline_low = splines[1]

        y_val_up = spline_up(x_val)
        y_val_low = spline_low(x_val)
        print(f'  y_up = {y_val_up.round(5)} and y_low = {y_val_low.round(5)}.')

        plt.figure(1)
        plt.plot(x_val, y_val_up, marker='o', color='orangered')
        plt.plot(x_val, y_val_low, marker='o', color='orangered')
        plt.axvline(x=x_val, color='orangered')

    plt.show()


class Profile:
    """Represents the profile of an aerofoil given from a set of datapoints. Fits a cubic spline and re-evaluates the
    datapoints as necessary for precision and consistency.

    Attributes
    ----------
    TODO: add attribute documentation
    """

    # Static dataframe variable
    dataframe = pd.DataFrame(columns=['name', 'x', 'y_up', 'y_low', 'spline'])

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
                new_row = pd.DataFrame({'name': [self.file], 'x': 'Error', 'y_up': 'Error', 'y_low': 'Error',
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
                new_row = pd.DataFrame({'name': [self.file], 'x': 'Error', 'y_up': 'Error', 'y_low': 'Error',
                                        'spline': 'Error'})
                Profile.dataframe = pd.concat([Profile.dataframe, new_row])
                return

    def update_dataframe(self):
        splinef_up = self.spline_funcs[0]
        splinef_low = self.spline_funcs[1]

        new_row = pd.DataFrame({'name': [self.name], 'x': [splinef_up[0]], 'y_up': [splinef_up[1]],
                                'y_low': [splinef_low[1]], 'spline': [self.splines]})
        Profile.dataframe = pd.concat([Profile.dataframe, new_row])

    def coord_profile(self):
        # Read aerofoil .dat file into dat list
        with open(self.file, "r") as f:
            lines = f.readlines()
            dat = []
            empty_line_indxs = []
            indx = 0
            for line in lines:
                line = line.strip()
                if line == '': empty_line_indxs.append(indx)
                dat.append(line)
                indx += 1

        # Set aerofoil name
        if dat[0][-8:] == ' AIRFOIL':
            name = dat[0][0:-8]
        else:
            name = dat[0]
        if dat[1] == '' or dat[2] == '':
            dat = dat[empty_line_indxs[0] + 1:]
        else:
            dat = dat[1:]

        # Float dat list
        dat_flt = []
        for line in dat:
            line = line.split()
            line = [float(pt) for pt in line]
            dat_flt.append(line)

        if len(empty_line_indxs) == 0 or len(empty_line_indxs) == 1:
            xs = [coord[0] for coord in dat_flt]
            ys = [coord[1] for coord in dat_flt]
            indx = 1
            for x in xs[1:]:
                d = xs[indx + 1] - x
                if d > 0:
                    xmin_indx = indx
                    xs.insert(xmin_indx, x)
                    ys.insert(xmin_indx, ys[xmin_indx])
                    break
                indx += 1
            xs_up = xs[:xmin_indx + 1]
            xs_low = xs[xmin_indx + 1:]
            ys_up = ys[:xmin_indx + 1]
            ys_low = ys[xmin_indx + 1:]

        else:
            coords_up = dat_flt[:empty_line_indxs[1] - empty_line_indxs[0] - 1]
            xs_up = [coord[0] for coord in coords_up]
            ys_up = [coord[1] for coord in coords_up]

            coords_low = dat_flt[empty_line_indxs[1] - empty_line_indxs[0]:]
            xs_low = [coord[0] for coord in coords_low]
            ys_low = [coord[1] for coord in coords_low]

        # Sort upper and lower in ascending x
        sorter = pd.DataFrame({'x': xs_up, 'y': ys_up})
        sorter = sorter.sort_values('x')
        xs_up = list(sorter.x)
        ys_up = list(sorter.y)
        sorter = pd.DataFrame({'x': xs_low, 'y': ys_low})
        sorter = sorter.sort_values('x')
        xs_low = list(sorter.x)
        ys_low = list(sorter.y)

        coords_up = [xs_up, ys_up]
        coords_low = [xs_low, ys_low]

        # Change duplicates in xs
        # TODO

        xs_low_rev = list(xs_low)
        xs_low_rev.reverse()
        x = xs_up + xs_low_rev
        ys_low_rev = list(ys_low)
        ys_low_rev.reverse()
        y = ys_up + ys_low_rev

        self.name = name
        self.coords_up = coords_up
        self.coords_low = coords_low
        self.x = x
        self.y = y
        return name, coords_up, coords_low, x, y

    def x_distribution(self):
        spline_xs_lin = np.linspace(0.0, math.pi, self.points)
        spline_xs = 0.5 * (1 - np.cos(spline_xs_lin))

        self.spline_xs = spline_xs
        return spline_xs

    def get_spline(self):
        spline_up = interp.InterpolatedUnivariateSpline(self.coords_up[0], self.coords_up[1])
        yfunc_up = spline_up(self.spline_xs)
        spline_func_up = [self.spline_xs.round(5), yfunc_up.round(5)]

        spline_low = interp.InterpolatedUnivariateSpline(self.coords_low[0], self.coords_low[1])
        yfunc_low = spline_low(self.spline_xs)
        spline_func_low = [self.spline_xs.round(5), yfunc_low.round(5)]

        splines = [spline_up, spline_low]
        spline_funcs = [spline_func_up, spline_func_low]

        self.spline_funcs = spline_funcs
        self.splines = splines
        return splines, spline_funcs
