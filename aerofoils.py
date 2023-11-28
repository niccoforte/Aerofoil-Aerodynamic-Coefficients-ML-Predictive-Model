import os
import re
from bs4 import BeautifulSoup
import urllib.request as urllib2

import numpy as np
import math
import scipy.interpolate as interp
from matplotlib import pyplot as plt
import pandas as pd


def get_UIUC_foils(directory='dat/aerofoil-dat'):
    """Achieves all aerofoil coordinate files from the UIUC Airfoil Coordinate Database and downloads them to a
    specified directory.

    Parameters
    ----------
    directory : str, default 'dat/aerofoil-dat'
        Direcotry path where to download the '.dat' coordinate files.
    """

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

    print(f'-Done. {indx} files copied from https://m-selig.ae.illinois.edu/ads/coord_database.html and saved to: '
          f'~/{directory}.')


def get_AFT_foils(directory='dat/aerofoil-dat'):
    """Achieves all aerofoil coordinate files from the Airfoil Tools Airfoil Database and downloads them to a
    specified directory.

    Parameters
    ----------
    directory : str, default 'dat/aerofoil-dat'
        Direcotry path where to download the '.dat' coordinate files.
    """

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

    print(f'-Done. {indx} files copied from http://airfoiltools.com/search/airfoils and saved to: ~/{directory}.')


def get_RENNES_foils(directory='dat/rennes-dat/aerofoil-dat'):
    """Achieves all aerofoil coordinate files from a study conducted by the Université de Rennes and downloads them to
    a specified directory.

    Parameters
    ----------
    directory : str, default 'dat/rennes-dat/aerofoil-dat'
        Direcotry path where to download the '.dat' coordinate files.
    """

    baseFlpth = "https://perso.univ-rennes1.fr/laurent.blanchard/Profils/"

    html_all = urllib2.urlopen(baseFlpth).read()
    soup_all = BeautifulSoup(html_all, 'html.parser')

    links_all = [link['href'] for link in soup_all.find_all('a', href=re.compile("/index"))]
    links_all.remove('centrepoussee/index.html')
    links_all.remove('clouet/index.html')

    linknames = [link[:-11].lower() for link in links_all]
    filenames_dir = [str(file)[11:-6].lower() for file in os.scandir(directory)]
    linknames_new = [name for name in linknames if name not in filenames_dir]
    links_new = []
    for new in linknames_new:
        for link in links_all:
            if new in link:
                links_new.append(link)

    print('Staring RENNES Aerofoil download...')
    indx = 0
    for link in links_new:
        try:
            html_foil = urllib2.urlopen(baseFlpth + link).read()
            soup_foil = BeautifulSoup(html_foil, 'html.parser')

            links_dat1 = [link_Re['href'] for link_Re in soup_foil.find_all('a', href=re.compile('.dat'))]
            links_dat2 = [link_Re['href'] for link_Re in soup_foil.find_all('a', href=re.compile('.DAT'))]
            links_dat = links_dat1 + links_dat2
            links_dat = [link for link in links_dat if link[:-4] not in filenames_dir]

            for link_dat in links_dat:
                fullfilename = os.path.join(directory, link_dat.lower())
                urllib2.urlretrieve(baseFlpth + link[:-10] + link_dat, fullfilename)

                indx += 1

        except:
            pass

    print(f'-Done. {indx} files copied from {baseFlpth} and saved to ~/{directory}.')


def create_profiles(directory='dat/aerofoil-dat', ext='dat', points=51, prnt=False):
    """Generates Profile objects from a directory of aerofoil coordinate files and returns a dictionary of aerofoil
     names to objects and a Pandas DataFrame of object attibutes.

    Parameters
    ----------
    directory : str, default 'dat/aerofoil-dat'
        Directory path that contains the aerofoil geometry coordinate files.
    ext : str, default 'dat'
        Extension of the relevant files in the directory.
    points : int, default 51
        Number of cosine-spaced points used to re-create the upper and lower aerofoil profile surfaces.
    prnt : bool, default False
        Print log as profile objects are created.

    Returns
    -------
    profiles : dict
        Dictionary of aerofoil profile names and objects.
    aerofoils_df : pandas.DataFrame
        DataFrame of names, filenames, x, y_up, y_low coordinates, spline functions, and xy_profiles for profiles
        objects.
    """

    if directory == 'dat/aerofoil-dat':
        print('Creating Aerofoils DataFrame...')
    elif directory == 'dat/rennes-dat/aerofoil-dat':
        print('Creating Rennes Aerofoils DataFrame...')

    aerofoils_df = pd.DataFrame(columns=['name', 'file', 'x', 'y_up', 'y_low', 'spline', 'xy_profile'])
    profiles = {}
    for file in os.scandir(directory):
        if file.name.endswith('.' + ext):
            try:
                p = Profile(file, points=points, prnt=prnt)
                file_name = str(p.file)
                profiles[file_name] = p

                new_row = pd.DataFrame({'name': [p.name], 'file': [p.file], 'x': [p.spline_funcs[0][0]],
                                        'y_up': [p.spline_funcs[0][1]], 'y_low': [p.spline_funcs[1][1]],
                                        'spline': [p.splines], 'xy_profile': [p.xy_profile]})
                aerofoils_df = pd.concat([aerofoils_df, new_row], ignore_index=True)

            except Exception as e:
                print(' ', file.name, 'failed. ERROR:', e)

    print(f'-Done. DataFrame created successfully with {len(aerofoils_df)} aerofoil profiles.')

    return profiles, aerofoils_df


def plot_profile(df, indx, scatt=False, x_val=None, pltfig=1, ax=None, prnt=False):
    """Plots the profile of a given aerofoil.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the aerofoils' profiles.
    indx : int
        Index in df of the aerofoil to be plotted.
    scatt : bool, defalut False
        Includes scatterplot of original coordinates.
    x_val : float, optional
        x-value at which to evaluate upper and lower profile y-coordinates.
    pltfig : int, default 1
        Figure on which to plot when plotting multiple aerofoils at once.
    ax : matplotlib.pyplot.axes, optional
        Axes on which to plot when plotting on subplots.
    prnt : bool, default False
        Print log as profile is being plotted.
    """

    if prnt:
        print(f'Plotting {df.name[indx]} ...')

    if ax is None:
        fig = plt.figure(pltfig)
        ax = fig.subplots(1, 1)

    ax.set_title(df.name[indx].upper(), fontsize=15, fontname="Times New Roman", fontweight='bold')
    ax.plot(df.x[indx], df.y_low[indx])
    ax.plot(df.x[indx], df.y_up[indx])
    ax.set_ylim([min(df.y_low[indx]) - 0.15, max(df.y_up[indx]) + 0.15])
    ax.grid()

    if scatt:
        if prnt:
            print(' Including scatter of original x - y coordinates.')

        ax.scatter(df.xy_profile[indx][0], df.xy_profile[indx][1], s=5)

    if x_val:
        print(f' Evaluating {df.name[indx]} y coordinates at x={x_val}...')

        splines = df.spline[indx]
        spline_up = splines[0]
        spline_low = splines[1]

        y_val_up = spline_up(x_val)
        y_val_low = spline_low(x_val)
        print(f' y_up = {y_val_up.round(5)} and y_low = {y_val_low.round(5)}.')

        ax.plot(x_val, y_val_up, marker='o', color='orangered')
        ax.plot(x_val, y_val_low, marker='o', color='orangered')
        ax.axvline(x=x_val, color='orangered')

    plt.show()


def aerofoil_difference(df, name1, name2, plot=False):
    """Evaluates the difference between two aerofoil profile geometries.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the aerofoils' profiles.
    name1 : str
        Filename of first aerofoil profile.
    name2 : str
        Filename of second aerofoil profile.
    plot : bool, default False
        Plot profiles of aerofoils between which difference is evaluated.

    Returns
    -------
    d : float
        Sum of the absolute difference between all corresponding coordinates of the two aerofoil profiles.
    """

    y_up1, y_low1 = df[df.file == name1].y_up.tolist()[0], df[df.file == name1].y_low.tolist()[0]
    y_up2, y_low2 = df[df.file == name2].y_up.tolist()[0], df[df.file == name2].y_low.tolist()[0]

    du_tip = 0
    dl_tip = 0
    for u1, l1, u2, l2 in zip(y_up1[:23], y_low1[:23], y_up2[:23], y_low2[:23]):
        _du = u2 - u1
        _dl = -(l2 - l1)
        du_tip += _du
        dl_tip += _dl
    d_tip = du_tip - dl_tip

    du_tail = 0
    dl_tail = 0
    for u1, l1, u2, l2 in zip(y_up1[23:], y_low1[23:], y_up2[23:], y_low2[23:]):
        _du = u2 - u1
        _dl = -(l2 - l1)
        du_tail += _du
        dl_tail += _dl
    d_tail = du_tail - dl_tail

    d = abs(du_tip) + abs(dl_tip) + abs(du_tail) + abs(dl_tail)

    if plot:
        ind1 = df[df.file == name1].index[0]
        plot_profile(df, ind1, scatt=False, x_val=None, pltfig=1, prnt=False)
        ind2 = df[df.file == name2].index[0]
        plot_profile(df, ind2, scatt=False, x_val=None, pltfig=2, prnt=False)

    return d


class Profile:
    """Represents the profile of an aerofoil given from a set of datapoints. Fits a smooth spline and re-evaluates the
    datapoints as necessary for precision and consistency.

    Parameters
    ----------
    file : str
        Path to aerofoil coordinate file.
    points : int, default 51
        Number of cosine-spaced points used to recreate the upper and lower aerofoil profile surfaces.
    prnt : bool, default False
        Print log as profile objects are created.

    Attributes
    ----------
    name : str
        Aerofoil name.
    file : str
        Aerofoil filename.
    coords_up : list
        Upper profile surface raw x and y coordinates.
    coords_low : list
        Lower profile surface raw x and y coordinates.
    x : list
        Full set of raw upper and lower profile x coordinates.
    y : list
        Full set of raw upper and lower profile y coordinates.
    xy_profile : list
        x and y united into one list.
    spline_xs : list
        Cosine spaced x-ccordinates at which to re-create raw coordinates for each profile surface.
        Number of x-coordinates defined in 'points' parameter.
    splines : list
        scipy.interpolate.Akima1DInterpolator interplolative spline functions for upper and lower profiles.
    spline_funcs : list
        Upper and lower profile cosine spaced interpolated x and y coordinates.
    """

    def __init__(self, file, points=51, prnt=False):
        self.file = file
        self.points = points

        self.name = None
        self.coords_up = []
        self.coords_low = []
        self.x = []
        self.y = []
        self.xy_profile = []

        self.spline_xs = []

        self.splines = []
        self.spline_funcs = []

        if prnt:
            print('Achieving profile coordinates...')
            self.name, self.file, self.coords_up, self.coords_low, self.x, self.y, self.xy_profile = self.coord_profile()
            print(f' {self.name} Done. Creating x-coordinate cosine distribution with {points} points...')
            self.spline_xs = self.x_distribution()
            print(f'  Done. Interpolating upper and lower profile splines at coordinates...')
            self.splines, self.spline_funcs = self.get_spline()
            print('   Done')
        else:
            self.name, self.file, self.coords_up, self.coords_low, self.x, self.y, self.xy_profile = self.coord_profile()
            self.spline_xs = self.x_distribution()
            self.splines, self.spline_funcs = self.get_spline()

    def coord_profile(self):
        """Reads coordinate files, separates upper and lower profiles, scales leading and trailing edge between (0, 0)
        and (1, 0), and prepares corrdinates for interpolative function.

        Returns
        -------
        name : str
            Aerofoil name.
        file : str
            Aerofoil filename.
        coords_up : list
            Upper profile surface raw x and y coordinates.
        coords_low : list
            Lower profile surface raw x and y coordinates.
        x : list
            Full set of raw upper and lower profile x coordinates.
        y : list
            Full set of raw upper and lower profile y coordinates.
        xy_profile : list
            x and y united into one list.
        """

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

        # Set aerofoil name, change file, cleanup header rows, & remove empty final rows
        name = dat[0]
        name = name.replace('AIRFOIL', 'Aerofoil')

        file = str(self.file)[11:-6].lower().replace('.', '').replace('_', '-')

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

        # Remove x duplicates by neglibibly increasing second.
        for indx, x in enumerate(xs_up[:-1]):
            d = xs_up[indx + 1] - x
            if d > 0:
                pass
            elif d == 0:
                xs_up[indx + 1] = xs_up[indx + 1] + 0.00002
            elif d < 0:
                break

        for indx, x in enumerate(xs_low[:-1]):
            d = xs_low[indx + 1] - x
            if d > 0:
                pass
            elif d == 0:
                xs_low[indx + 1] = xs_low[indx + 1] + 0.00002
            elif d < 0:
                break

        # Remove outliers in x domain (x>5).
        for indx, x in enumerate(xs_up):
            if x > 5:
                xs_up.remove(x)
                xs_up.insert(indx, max(xs_up) + 0.00002)
        for indx, x in enumerate(xs_low):
            if x > 5:
                xs_low.remove(x)
                xs_low.insert(indx, max(xs_low) + 0.00002)

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

        return name, file, coords_up, coords_low, x, y, xy_profile

    def x_distribution(self):
        """Creates a set of cosine spaces x-coordinates between 0 and 1. Quantity defined by 'points' parameter.

        Returns
        -------
        spline_xs : list
            Cosine spaced x-ccordinates at which to re-create raw coordinates for each profile surface.
            Number of x-coordinates defined in 'points' parameter.
        """

        spline_xs_lin = np.linspace(0.0, math.pi, self.points)
        spline_xs = 0.5 * (1 - np.cos(spline_xs_lin))

        return spline_xs

    def get_spline(self):
        """Re-creates upper and lower profiles by evaluating their y-coordinates at the cosine spaced x-coordinates
        defines by the x_distribution() function.

        Returns
        -------
        splines : list
            scipy.interpolate.Akima1DInterpolator interplolative spline functions for upper and lower profiles.
        spline_funcs : list
            Upper and lower profile cosine spaced interpolated x and y coordinates.
        """

        spline_up = interp.Akima1DInterpolator(self.coords_up[0], self.coords_up[1])
        yfunc_up = spline_up(self.spline_xs)
        spline_func_up = [self.spline_xs, yfunc_up]

        spline_low = interp.Akima1DInterpolator(self.coords_low[0], self.coords_low[1])
        yfunc_low = spline_low(self.spline_xs)
        spline_func_low = [self.spline_xs, yfunc_low]

        splines = [spline_up, spline_low]
        spline_funcs = [spline_func_up, spline_func_low]

        return splines, spline_funcs
