from matplotlib import pyplot as plt
import pandas as pd

import profile

# Create list of profile objects
profiles = profile.create_profiles()


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
    plt.plot(df.x[indx], df.y_high[indx])
    plt.ylim([min(df.y_low[indx]) - 0.15, max(df.y_high[indx]) + 0.15])
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


aerofoils_df = profile.Profile.dataframe
# plot_profile(aerofoils_df, 4, x_val=0.38)
# aerofoils_df

worked = aerofoils_df[aerofoils_df['x'] != 'Error'].reset_index()
plot_profile(worked, 72, x_val=0.36)
# len(worked.name.tolist()), worked.name.tolist()
