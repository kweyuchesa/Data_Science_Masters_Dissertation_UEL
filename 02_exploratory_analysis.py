# ========================== 2. EXPLORATORY DATA ANALYSIS ==========================
print("\n--Exploratory Data Analysis --")

# Trade flows by pair
pair_trade = df.groupby(['countryi', 'countryj'])['exp_ij'].sum().reset_index()
print(pair_trade)

# Plot trade trends
plt.figure(figsize=(14, 7))
for pair, group in df.groupby(['countryi', 'countryj']):
    plt.plot(group['time'], group['exp_ij'], label=f"{pair[0]}-{pair[1]}")
plt.title("Bilateral Export Flows Over Time (Northern Corridor)")
plt.xlabel("Time")
plt.ylabel("Export Value (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('trade_trends.png')
plt.show()

# Correlation heatmap (key variables)
key_vars = ['exp_ij_log', 'gdp_i', 'gdp_j', 'wdist_ij', 'tr_mbs', 'cdt_mbs', 
            'wbcnon_nc', 'ke_ntb_ug1', 'ke_ntb_eac1', 'rw_ntb_ke']
corr = df[key_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png')
plt.show()
