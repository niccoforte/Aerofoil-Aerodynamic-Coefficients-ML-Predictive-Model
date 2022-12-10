import pandas as pd
import random
import profile

# Download aerofoil .dat files to 'aerofoil_dat' directory
# profile.get_aerofoils()

# Create list and dataframe of profile objects
profiles, aerofoils_df = profile.create_profiles()

pd.set_option('display.max_columns', None)
print(aerofoils_df)

# profile.plot_profile(aerofoils_df, 946, scatt=True, x_val=None, pltfig=1)

# for i in range(10):
#     r = random.randint(0, len(aerofoils_df))
#     profile.plot_profile(aerofoils_df, r, scatt=True, x_val=None, pltfig=i)
#     print(r)