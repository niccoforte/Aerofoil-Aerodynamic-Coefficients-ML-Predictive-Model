import profile

# Download aerofoil .dat files to 'aerofoil_dat' directory
# profile.get_aerofoils()

# Create list and dataframe of profile objects
profiles, aerofoils_df = profile.create_profiles()

worked = aerofoils_df[aerofoils_df['x'] != 'Error'].reset_index()
# plot_profile(worked, 61, x_val=None, pltfig=1)

not_worked = aerofoils_df[aerofoils_df['x'] == 'Error'].reset_index()
print(len(not_worked.name.tolist()), not_worked.name.tolist())