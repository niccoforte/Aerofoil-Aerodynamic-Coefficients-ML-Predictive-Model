from matplotlib import pyplot as plt
import pandas as pd

import profile


# Create list and dataframe of profile objects
profiles, aerofoils_df = profile.create_profiles()

worked = aerofoils_df[aerofoils_df['x'] == 'Error'].reset_index()
# profile.plot_profile(worked, 726, x_val=0.36)
print(len(worked.name.tolist()), worked.name.tolist())