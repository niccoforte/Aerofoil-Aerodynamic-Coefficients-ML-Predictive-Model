from matplotlib import pyplot as plt
import pandas as pd
import os

import profile


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
            p = profile.Profile(file, points=51, prnt=False)

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

    plt.show()


aerofoils_df = splines_df()
# plot_profile(aerofoils_df, 4, x_val=0.38)
# aerofoils_df

worked = aerofoils_df[aerofoils_df['Xs'] != 'Error'].reset_index()
plot_profile(worked, 72, x_val=0.36)
# len(worked.name.tolist()), worked.name.tolist()
