# ========================== 3. FEATURE ENGINEERING ==========================
print("\n--Feature Engineering --")

# Create interaction terms and logs (Gravity Theory)
df['log_gdp_i'] = np.log(df['gdp_i'])
df['log_gdp_j'] = np.log(df['gdp_j'])
df['log_dist'] = np.log(df['wdist_ij'])
df['log_pop_i'] = np.log(df['pop_i'])
df['log_pop_j'] = np.log(df['pop_j'])

# Economic mass
df['gdp_product'] = df['gdp_i'] * df['gdp_j']
df['log_gdp_product'] = np.log(df['gdp_product'])

# NTB count / intensity
df['ntb_count'] = df[[col for col in df.columns if 'ntb' in col]].sum(axis=1)

# Time fixed effects
df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

print("Engineered features created.")
