import aerofoils
import cases
import data
import pandas as pd
import random


# Download aerofoil .dat files to 'aerofoil_dat' directory and case .csv files to 'case_dat' directory.
aerofoils.get_UIUC_foils(directory='aerofoil_dat')
aerofoils.get_AFT_foils(directory='aerofoil_dat')
cases.get_AFT_cases(directory='case_dat')


# Create dictionary of Profile objects and Aerofoils DataFrame.
profiles, aerofoils_df = aerofoils.create_profiles(points=51, prnt=False)


# Create DataFrame of case data.
cases_df = cases.create_cases()


# Merge aerofoils and cases dataframes.
data_df = data.merge(aerofoils_df, cases_df)




# ----- Print data_df

#pd.set_option('display.max_columns', None)
#print(data_df)


# ----- Aerofoil Plots

# profile.plot_profile(aerofoils_df, 945, scatt=True, x_val=0.004, pltfig=1)

# for i in range(10):
#     r = random.randint(0, len(aerofoils_df))
#     profile.plot_profile(aerofoils_df, r, scatt=True, x_val=None, pltfig=i)
#     print(r)
