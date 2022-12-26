import pandas as pd
import random
import aerofoils
import cases

# Download aerofoil .dat files to 'aerofoil_dat' directory and case .csv files to 'case_dat' directory
# profile.get_aerofoils()
# cases.get_cases()

# Create dictionary of Profile objects and Aerofoils DataFrame
profiles, aerofoils_df = aerofoils.create_profiles(k=3, points=51, prnt=False)

# Create DataFrame of case data
cases_df = cases.create_cases()

# Merge aerofoils and cases dataframes
data_df = pd.merge(aerofoils_df, cases_df, on="file")

pd.set_option('display.max_columns', None)
print(data_df)


# GRAPHS BELOW

# profile.plot_profile(aerofoils_df, 945, scatt=True, x_val=0.004, pltfig=1)

# for i in range(10):
#     r = random.randint(0, len(aerofoils_df))
#     profile.plot_profile(aerofoils_df, r, scatt=True, x_val=None, pltfig=i)
#     print(r)
