# ========================== 6. BEST MODEL & SHAP INTERPRETABILITY ==========================
print("\n-- Training Best Model & SHAP Analysis --")

# Train final XGBoost (usually best performer)
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', models['XGBoost'])
])
final_model.fit(X, y)

# SHAP values
explainer = shap.TreeExplainer(final_model.named_steps['regressor'])
X_processed = final_model.named_steps['preprocessor'].transform(X)
shap_values = explainer.shap_values(X_processed)

# Summary plot
shap.summary_plot(shap_values, X_processed, feature_names=preprocessor.get_feature_names_out(), 
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Trade Flow Drivers")
plt.tight_layout()
plt.savefig('shap_importance.png')
plt.show()

# Dependence plots for key NTBs
shap.dependence_plot("ntb_count", shap_values, X_processed, 
                    feature_names=preprocessor.get_feature_names_out(), show=False)
plt.savefig('shap_ntb_dependence.png')
plt.show()

print("SHAP analysis completed.")

# ========================== 7. MODEL COMPARISON & EXPORT ==========================
comparison = pd.DataFrame(results).T
print("\n=== Model Comparison ===")
print(comparison.round(4))

# Save results
comparison.to_csv('model_comparison.csv')
df.to_csv('processed_trade_data.csv', index=False)

print("\n Full analysis pipeline completed successfully!")
print("Files saved: trade_trends.png, correlation_matrix.png, shap_*.png, model_comparison.csv")
