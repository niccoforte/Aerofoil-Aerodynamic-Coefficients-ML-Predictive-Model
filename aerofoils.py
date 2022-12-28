import os
import re
import math
import urllib.request as urllib2
import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup


def get_UIUC_foils(directory='aerofoil_dat'):
    """Achieve all '.dat' files from UIUC aerofoil coordinate database and downloads to directory."""

    baseFlpth = "https://m-selig.ae.illinois.edu/ads/"

    html_page = urllib2.urlopen("https://m-selig.ae.illinois.edu/ads/coord_database.html")
    soup = BeautifulSoup(html_page, 'lxml')

    links_all = [link['href'] for link in soup.find_all('a', href=re.compile('\.dat', re.IGNORECASE))]

    linknames = []
    for link in links_all:
        if link.startswith('coord_updates/'):
            linknames.append(link[14:-4].lower())
        elif link.startswith('coord/'):
            linknames.append(link[6:-4].lower())
    filenames_dir = [str(file)[11:-6].lower() for file in os.scandir(directory)]
    linknames_new = [name for name in linknames if name not in filenames_dir]
    links_new = []
    for new in linknames_new:
        for link in links_all:
            if new in link:
                links_new.append(link)

    print('Starting UIUC Aerofoil download...')
    indx = 0
    for link in links_new:
        fullfilename = os.path.join(directory, link.rsplit('/')[-1])
        urllib2.urlretrieve(baseFlpth + link, fullfilename)

        indx += 1

    print(f' Done. {indx} files copied from https://m-selig.ae.illinois.edu/ads/coord_database.html and saved to: '
          f'~/{directory}.')


def get_AFT_foils(directory='aerofoil_dat'):
    baseFlpth = "http://airfoiltools.com"

    html_all = urllib2.urlopen("http://airfoiltools.com/search/airfoils").read()
    soup_all = BeautifulSoup(html_all, 'html.parser')

    links_all = [link['href'] for link in soup_all.find_all('a', href=re.compile("/airfoil/details", re.IGNORECASE))]

    linknames = [link[25:-3].lower() for link in links_all]
    filenames_dir = [str(file)[11:-6].lower() for file in os.scandir(directory)]
    linknames_new = [name for name in linknames if name not in filenames_dir]
    links_new = []
    for new in linknames_new:
        for link in links_all:
            if new in link:
                links_new.append(link)

    print('Starting AFT Aerofoil download...')
    indx = 0
    for link in links_new:
        html_foil = urllib2.urlopen(baseFlpth + link).read()
        soup_foil = BeautifulSoup(html_foil, 'html.parser')

        link_dat = soup_foil.find_all('a', href=re.compile("/airfoil/lednicerdatfile"))[0]['href']
        name = link_dat[33:-3] + '.dat'

        fullfilename = os.path.join(directory, name)
        urllib2.urlretrieve(baseFlpth + link_dat, fullfilename)

        indx += 1

    print(f' Done. {indx} files copied from http://airfoiltools.com/search/airfoils and saved to: ~/{directory}.')


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

    print('Creating Aerofoils DataFrame...')

    aerofoils_df = pd.DataFrame(columns=['name', 'file', 'x', 'y_up', 'y_low', 'spline', 'xy_profile'])
    profiles = {}
    for file in os.scandir(directory):
        if file.name.endswith('.' + ext):
            try:
                p = Profile(file, points=points, prnt=prnt)
                file_name = str(p.file)

                profiles[file_name[11:-2]] = p

                new_row = pd.DataFrame(
                    {'name': [p.name], 'file': [p.file], 'x': [p.spline_funcs[0][0]], 'y_up': [p.spline_funcs[0][1]],
                     'y_low': [p.spline_funcs[1][1]], 'spline': [p.splines], 'xy_profile': [p.xy_profile]})
                aerofoils_df = pd.concat([aerofoils_df, new_row], ignore_index=True)

            except Exception as e:
                print('', file.name, 'failed. Error:', e)

    print(' Aerofoils DataFrame created successfully.')

    return profiles, aerofoils_df


def plot_profile(df, indx, scatt=False, x_val=None, pltfig=1):
    """Plots the profile of a given aerofoil.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the aerofoils' data.
    indx : int
        Index of the aerofoil to be plotted.
    scatt : bool or None
        Includes scatterplot of original coordinates.
    x_val : float, optional
        x-value to be plotted along with the aerofoil.
    pltfig : int
        Figure on which to plot when plotting multiple aerofoils at once.
    """

    print(f'Plotting {df.name[indx]} Aerofoil...')

    plt.figure(pltfig)
    plt.title(df.name[indx])
    plt.plot(df.x[indx], df.y_low[indx])
    plt.plot(df.x[indx], df.y_up[indx])
    plt.ylim([min(df.y_low[indx]) - 0.15, max(df.y_up[indx]) + 0.15])
    plt.grid()

    if scatt:
        print(' Including scatter of original x - y coordinates.')
        plt.scatter(df.xy_profile[indx][0], df.xy_profile[indx][1], s=10)

    if x_val:
        print(f' Evaluating y-coordinates at x={x_val}...')

        splines = df.spline[indx]
        spline_up = splines[0]
        spline_low = splines[1]

        y_val_up = spline_up(x_val)
        y_val_low = spline_low(x_val)
        print(f'  y_up = {y_val_up.round(5)} and y_low = {y_val_low.round(5)}.')

        plt.figure(pltfig)
        plt.plot(x_val, y_val_up, marker='o', color='orangered')
        plt.plot(x_val, y_val_low, marker='o', color='orangered')
        plt.axvline(x=x_val, color='orangered')

    plt.show()


class Profile:
    """Represents the profile of an aerofoil given from a set of datapoints. Fits a cubic spline and re-evaluates the
    datapoints as necessary for precision and consistency.

    Attributes
    ----------
    file : .dat
        UIUC aerofoil coordinate file.
    points : int
        Points at which to evaluate the splines of the aerofoils.
    prnt : bool
        Print log as profile objects are created.
    """

    def __init__(self, file, points=51, prnt=False):
        self.file = file
        self.points = points

        self.name = None
        self.coords_up = None
        self.coords_low = None
        self.x = None
        self.y = None
        self.xy_profile = None

        self.spline_xs = None

        self.splines = None
        self.spline_funcs = None

        if prnt:
            print('Achieving profile coordinates...')
            self.coord_profile()
            print(f' {self.name} Done. Creating x-coordinate cosine distribution with {points} points...')
            self.x_distribution()
            print(f'  Done. Interpolating upper and lower profile splines at coordinates...')
            self.get_spline()
            print('   Done')
        else:
            self.coord_profile()
            self.x_distribution()
            self.get_spline()

    def coord_profile(self):
        # Read aerofoil .dat file into dat list
        with open(self.file, "r") as f:
            lines = f.readlines()
            dat = []
            empty_line_indxs = []
            for indx, line in enumerate(lines):
                line = line.strip()
                if line == '':
                    empty_line_indxs.append(indx)
                dat.append(line)

        # Set aerofoil name, change filename, cleanup header rows, & remove empty final rows
        name = dat[0]
        name = name.replace('AIRFOIL', 'Aerofoil')

        file = str(self.file)[11:-6].lower().replace('.','').replace('_','-')

        if dat[2] == '':
            dat = dat[empty_line_indxs[0] + 1:]
        else:
            dat = dat[1:]

        if dat[-1] == '' or dat[-1] == '\n':
            dat = dat[:-1]
        else:
            pass

        # Float dat list
        dat_flt = []
        if dat[-1] == '\x1a' or '\x1a25' in dat[-1]:
            dat = dat[:-1]
            for line in dat:
                line = line.split()
                line = [float(pt) for pt in line]
                dat_flt.append(line)
        else:
            for line in dat:
                line = line.replace('......', '0.0').replace('(', '').replace(')', '')
                line = line.split()
                line = [float(pt) for pt in line]
                dat_flt.append(line)

        # Split upper and lower coordinates for different file types
        try:
            coords_up = dat_flt[:empty_line_indxs[1] - empty_line_indxs[0] - 1]
            xs_up = [coord[0] for coord in coords_up]
            ys_up = [coord[1] for coord in coords_up]

            coords_low = dat_flt[empty_line_indxs[1] - empty_line_indxs[0]:]
            xs_low = [coord[0] for coord in coords_low]
            ys_low = [coord[1] for coord in coords_low]

        except:
            dat_flt = list(filter(None, dat_flt))
            xs = [coord[0] for coord in dat_flt]
            ys = [coord[1] for coord in dat_flt]
            for indx, x in enumerate(xs[:-1]):
                d = xs[indx + 1] - x
                if d < 0:
                    pass
                elif d == 0:
                    d1 = xs[indx + 2] - xs[indx + 1]
                    if d1 == 0:
                        xmin_indx = indx + 1
                        xs.insert(xmin_indx, x)
                        ys.insert(xmin_indx, ys[xmin_indx])
                        break
                    elif d1 > 0:
                        xmin_indx = indx
                        break
                    elif d1 < 0:
                        pass
                elif d > 0:
                    xmin_indx = indx
                    xs.insert(xmin_indx, x)
                    ys.insert(xmin_indx, ys[xmin_indx])
                    break
            xs_up = xs[:xmin_indx + 1]
            xs_low = xs[xmin_indx + 1:]
            ys_up = ys[:xmin_indx + 1]
            ys_low = ys[xmin_indx + 1:]

        # Sort upper and lower in ascending x
        sorter = pd.DataFrame({'x': xs_up, 'y': ys_up})
        sorter = sorter.sort_values('x')
        xs_up = list(sorter.x)
        ys_up = list(sorter.y)

        sorter = pd.DataFrame({'x': xs_low, 'y': ys_low})
        sorter = sorter.sort_values('x')
        xs_low = list(sorter.x)
        ys_low = list(sorter.y)

        # Remove x duplicates by slightly increasing second.
        for indx, x in enumerate(xs_up[:-1]):
            d = xs_up[indx + 1] - x
            if d > 0:
                pass
            elif d == 0:
                xs_up[indx + 1] = xs_up[indx + 1] + 0.000025
            elif d < 0:
                break
        for indx, x in enumerate(xs_low[:-1]):
            d = xs_low[indx + 1] - x
            if d > 0:
                pass
            elif d == 0:
                xs_low[indx + 1] = xs_low[indx + 1] + 0.000025
            elif d < 0:
                break

        # Remove outliers in x domain (x>5).
        for indx, x in enumerate(xs_up):
            if x > 5:
                xs_up.remove(x)
                xs_up.insert(indx, max(xs_up) + 0.000025)
        for indx, x in enumerate(xs_low):
            if x > 5:
                xs_low.remove(x)
                xs_low.insert(indx, max(xs_low) + 0.000025)

        # Set domain between 0 and 1.
        if min(xs_up) < 0:
            xs_up = [x + abs(min(xs_up)) for x in xs_up]
        if min(xs_up) > 0:
            xs_up = [x - abs(min(xs_up)) for x in xs_up]
        if max(xs_up) < 1:
            ratio = 1 / max(xs_up)
            xs_up = [x * ratio for x in xs_up]
            ys_up = [y * ratio for y in ys_up]
        if max(xs_up) > 1:
            ratio = max(xs_up)
            xs_up = [x / ratio for x in xs_up]
            ys_up = [y / ratio for y in ys_up]

        if min(xs_low) < 0:
            xs_low = [x + abs(min(xs_low)) for x in xs_low]
        if min(xs_low) > 0:
            xs_low = [x - abs(min(xs_low)) for x in xs_low]
        if max(xs_low) < 1:
            ratio = 1 / max(xs_low)
            xs_low = [x * ratio for x in xs_low]
            ys_low = [y * ratio for y in ys_low]
        if max(xs_low) > 1:
            old_max = max(xs_low)
            xs_low = [x / old_max for x in xs_low]
            ys_low = [y / old_max for y in ys_low]

        # Create upper and lower pairs with both x and y coords
        coords_up = [xs_up, ys_up]
        coords_low = [xs_low, ys_low]

        # Create full continuous x and y lists for coord plotting
        xs_low_rev = list(xs_low)
        xs_low_rev.reverse()
        x = xs_up + xs_low_rev
        ys_low_rev = list(ys_low)
        ys_low_rev.reverse()
        y = ys_up + ys_low_rev
        xy_profile = [x, y]

        self.name = name
        self.file = file
        self.coords_up = coords_up
        self.coords_low = coords_low
        self.x = x
        self.y = y
        self.xy_profile = xy_profile
        return name, coords_up, coords_low, xy_profile

    def x_distribution(self):
        spline_xs_lin = np.linspace(0.0, math.pi, self.points)
        spline_xs = 0.5 * (1 - np.cos(spline_xs_lin))

        self.spline_xs = spline_xs
        return spline_xs

    def get_spline(self):
        spline_up = interp.Akima1DInterpolator(self.coords_up[0], self.coords_up[1])
        yfunc_up = spline_up(self.spline_xs)
        spline_func_up = [self.spline_xs, yfunc_up]

        spline_low = interp.Akima1DInterpolator(self.coords_low[0], self.coords_low[1])
        yfunc_low = spline_low(self.spline_xs)
        spline_func_low = [self.spline_xs, yfunc_low]

        splines = [spline_up, spline_low]
        spline_funcs = [spline_func_up, spline_func_low]

        self.spline_funcs = spline_funcs
        self.splines = splines
        return splines, spline_funcs
