# ========================== 5. MACHINE LEARNING PIPELINE ==========================
print("\n=== Machine Learning Models ===")

# Feature selection for ML
feature_cols = [
    'log_gdp_i', 'log_gdp_j', 'log_dist', 'log_pop_i', 'log_pop_j',
    'tr_mbs', 'cdt_mbs', 'wbcnon_nc', 'ntb_count',
    'contig', 'com_lang', 'landlci', 'landlcj',
    'ke_ntb_ug1', 'ke_ntb_eac1', 'rw_ntb_ke', 'ke_ntb_eac2'
]

X = df[feature_cols]
y = df['exp_ij']  # Use raw values for ML (better for tree models)

# Time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Preprocessing
numeric_features = [col for col in feature_cols if col not in ['contig', 'com_lang', 'landlci', 'landlcj']]
categorical_features = ['contig', 'com_lang', 'landlci', 'landlcj']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, 
                               subsample=0.8, colsample_bytree=0.8, random_state=42),
    'PPML_Baseline': None  # Already fitted
}

results = {}

for name, model in models.items():
    if name == 'PPML_Baseline':
        continue
        
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    mae_scores, rmse_scores, r2_scores = [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))
    
    results[name] = {
        'MAE': np.mean(mae_scores),
        'RMSE': np.mean(rmse_scores),
        'R2': np.mean(r2_scores)
    }
    
    print(f"{name:15} - MAE: {results[name]['MAE']:,.0f} | RMSE: {results[name]['RMSE']:,.0f} | R2: {results[name]['R2']:.4f}")
