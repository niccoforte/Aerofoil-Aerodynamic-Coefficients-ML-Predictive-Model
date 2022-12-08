import pandas as pd
import profile

# Download aerofoil .dat files to 'aerofoil_dat' directory
# profile.get_aerofoils()

# Create list and dataframe of profile objects
profiles, aerofoils_df = profile.create_profiles()

worked = aerofoils_df[aerofoils_df['x'] != 'Error']
not_worked = aerofoils_df[aerofoils_df['x'] == 'Error']

# pd.set_option('display.max_columns', None)
# print(worked)
profile.plot_profile(worked, 61, scatt=True, x_val=None, pltfig=1)

# print(len(not_worked.name.tolist()), not_worked.name.tolist())