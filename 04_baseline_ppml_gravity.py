# ========================== 4. BASELINE PPML MODEL ==========================
print("\n-- Baseline PPML Gravity Model --")

# PPML specification (structural gravity)
ppml_formula = """exp_ij ~ log_gdp_i + log_gdp_j + log_dist + 
                 tr_mbs + cdt_mbs + wbcnon_nc + ntb_count + 
                 contig + com_lang + landlci + landlcj"""

# For high-dimensional FE, we use a simpler version first (can extend with pygravity)
model_ppml = poisson(ppml_formula, data=df)
result = model_ppml.fit(disp=False)
print(result.summary())

# Predictions
df['ppml_pred'] = result.predict(df)

print("✅ PPML Baseline fitted.")
