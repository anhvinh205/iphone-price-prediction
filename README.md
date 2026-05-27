# iPhone Resale Price Prediction

A machine learning project to predict the resale price of used iPhones using ensemble learning techniques. This project includes exploratory data analysis (EDA), model training, evaluation, and an interactive Streamlit web application for price prediction.

## Project Overview

This repository contains:
- **Data Analysis** (`iphone.py`): Exploratory data analysis with visualizations
- **Model Training** (`iphonetrain.py`): Training multiple regression models
- **Web Application** (`app.py`): Interactive Streamlit app for price predictions

**Dataset**: 619 iPhone records with 17 features including condition, specifications, and market price.

---

## Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `Model` | Categorical | iPhone model (e.g., iPhone SE 3, iPhone 13 Pro) |
| `battery_health` | Numeric | Battery health percentage (50-100) |
| `storage` | Numeric | Storage capacity in GB (32, 64, 128, 256, 512) |
| `Months_since_release` | Numeric | Months since phone was released |
| `colour` | Categorical | Phone color (Black, White, Blue, etc.) |
| `price_at` | Numeric | Original purchase price in LKR |
| `Exchange_rate_1_USD_to_LKR` | Numeric | USD to LKR conversion rate |
| `battery_renew` | Boolean | Whether battery was replaced |
| `screen_replacement` | Boolean | Whether screen was replaced |
| `display_replacement` | Boolean | Whether display was replaced |
| `ios_updates` | Numeric | Number of iOS updates received |
| `availability` | Categorical | Available or phone-only |
| `screen_damages` | Categorical | Screen damage status |
| `backglass_damages` | Boolean | Back glass damage |
| `current_price(LKR)` | Numeric | **Target variable** - Current resale price in LKR |

---

## Project Structure

```
iphone price prediction/
├── data/
│   └── iphone_release.csv          # Training dataset
├── models/
│   ├── decision_tree.joblib        # Trained Decision Tree model
│   ├── random_forest.joblib        # Trained Random Forest model
│   ├── adaboost.joblib             # Trained AdaBoost model
│   └── gradient_boosting.joblib    # Trained Gradient Boosting model
├── results/
│   ├── metrics.csv                 # Model performance metrics
│   ├── comparison.png              # Model comparison visualization
│   ├── eda_overview.png            # EDA price distribution & correlation
│   └── eda_categorical.png         # Categorical feature analysis
├── iphone.py                       # EDA script
├── iphonetrain.py                  # Model training script
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip or conda package manager

### Setup

1. **Clone or download the repository**
   ```bash
   cd "iphone price prediction"
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Exploratory Data Analysis (EDA)

Run the EDA script to generate visualizations:

```bash
python iphone.py
```

**Output**:
- `results/eda_overview.png` - Price distribution and correlation heatmap
- `results/eda_categorical.png` - Boxplots by categorical features
- Console statistics (shape, columns, missing values, descriptive stats)

### 2. Model Training

Train all models and save them:

```bash
python iphonetrain.py
```

**Output**:
- `models/*.joblib` - Serialized trained models
- `results/metrics.csv` - Performance metrics (MAE, MSE, R²)
- `results/comparison.png` - Bar chart comparison of all models
- Console logs with training progress

### 3. Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501/`

**Features**:
- Select iPhone model, storage, color, and condition
- Adjust battery health, months since release, iOS updates
- Input original purchase price and exchange rate
- Specify hardware condition (replacements/damages)
- Get instant price prediction in LKR and USD

---

## Model Performance

### Model Comparison

Based on validation set (186 samples):

| Model | MAE | MSE | R² |
|-------|-----|-----|-----|
| **Decision Tree** | 5,782.90 | 63,005,051 | 0.97 |
| **Random Forest** | 4,989.00 | 49,751,705 | 0.98 ⭐ |
| **AdaBoost** | 15,766.91 | 365,465,262 | 0.84 |
| **Gradient Boosting** | 5,088.93 | 47,047,366 | 0.98 ⭐ |

**Best Model**: Random Forest & Gradient Boosting (R² = 0.98)
- These models explain 98% of variance in iPhone resale prices
- Mean prediction error ~5,000 LKR (~$15 USD)

---

## Feature Importance Analysis

### Top Predictive Features (by model):

**Random Forest**:
1. Model (35%) - iPhone variant is the strongest predictor
2. Storage (8%) - Higher storage increases price
3. Months since release (5%) - Older phones are cheaper
4. Battery health (3%) - Better condition = higher price

**Gradient Boosting**:
1. Model (50%) - Model type dominates
2. Months since release (10%) - Time depreciation
3. Storage (8%) - Capacity matters
4. Colour (3%) - Some colors more valuable

**Decision Tree**:
1. Model (95%) - Nearly all decision power from model
2. Storage (2%)
3. Battery health (<1%)

**AdaBoost**:
1. Months since release (30%) - Age is critical
2. Model (35%) - Model variety
3. Price at (10%) - Original price matters
4. Storage (10%)

**Key Insights**:
- iPhone **model** is the dominant factor across all models
- **Age of device** (months since release) significantly impacts depreciation
- **Storage capacity** has moderate impact (higher = more valuable)
- Device **condition features** (damages, replacements) have minimal direct impact but interact with other features
- **Battery health** percentage shows weak individual importance

---

## Data Preprocessing Pipeline

The project uses scikit-learn's `ColumnTransformer` for robust preprocessing:

### Numeric Features (4 features):
- `battery_health`, `storage`, `Months_since_release`, `Exchange_rate_1_USD_to_LKR`
- **Treatment**: Missing values imputed with median strategy

### Categorical Features (10 features):
- `Model`, `colour`, `battery_renew`, `screen_replacement`, `display_replacement`, `screen_damages`, `backglass_damages`, `availability`, `ios_updates`, `price_at`
- **Treatment**: 
  - Missing values filled with `"__MISSING__"` placeholder
  - Ordinal encoding with unknown value mapping (-1)

### Removed Features:
- `IMEI_number` - Unique identifier, not predictive
- `Release_Date` - Redundant with `Months_since_release`

---

## Training Details

### Configuration
- **Train/Validation Split**: 70/30 (433 training, 186 validation samples)
- **Random State**: 1 (reproducible results)
- **Feature Scaling**: Applied via SimpleImputer (median for numeric)

### Models Trained

1. **Decision Tree Regressor**
   - Hyperparameters: defaults, random_state=1
   - Strength: Fast, interpretable
   - Weakness: Can overfit

2. **Random Forest Regressor**
   - Hyperparameters: n_estimators=100, n_jobs=-1, random_state=1
   - Strength: Parallel processing, robust
   - Weakness: Black box

3. **AdaBoost Regressor**
   - Hyperparameters: random_state=1
   - Strength: Reduces bias
   - Weakness: Slower training

4. **Gradient Boosting Regressor**
   - Hyperparameters: random_state=1
   - Strength: Sequential error correction
   - Weakness: More parameters to tune

---

## Streamlit App Walkthrough

### Input Sections

**iPhone Information**:
- Model selection (iPhone SE 2 to iPhone 15 Pro Max)
- Storage: 64GB, 128GB, 256GB, 512GB, 1TB
- Color selection from 11 options
- Availability status

**Device Condition**:
- Battery health slider (50-100%)
- Screen replacement: Yes/No
- Display replacement: Yes/No
- Screen damage status
- Back glass damage: Yes/No
- Battery renewal: Yes/No
- iOS updates count (0-20)
- Months since release (1-60)
- Original purchase price in LKR
- USD-to-LKR exchange rate

### Output

- **Predicted Price**: Displayed in LKR with USD equivalent
- **Smart Suggestions**:
  - Low battery warning if health < 80%
  - Damage warning if device has physical issues

---

## Example Prediction

**Input**: iPhone 13 Pro (256GB, White)
- Battery health: 85%
- Months since release: 18
- iOS updates: 8
- Original price: 250,000 LKR
- No damage, no replacements
- Exchange rate: 323

**Output**: ~155,000 - 180,000 LKR (~$480-560 USD)
*(Actual prediction depends on trained model)*

---

## Dependencies

All required packages are listed in `requirements.txt`:

```
numpy              # Numerical computing
pandas             # Data manipulation
matplotlib         # Visualization
seaborn            # Statistical visualization
scikit-learn       # Machine learning models
joblib             # Model serialization
jupyter            # Interactive notebooks
ipykernel          # Jupyter kernel
streamlit          # Web app framework
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Results & Findings

### Key Metrics
- **Best R² Score**: 0.98 (Random Forest & Gradient Boosting)
- **Average MAE**: ~5,000 LKR (±$15)
- **MSE Range**: 47M - 365M LKR²
- **Training Time**: <2 minutes for all models

### Model Recommendations

✅ **Use Random Forest** if you need:
- Best balance of performance & interpretability
- Slightly smaller MSE
- Feature importance visualization

✅ **Use Gradient Boosting** if you need:
- Marginal R² improvement
- Sequential refinement of predictions
- Flexibility for hyperparameter tuning

❌ **Avoid AdaBoost** - Performance is notably worse (R² = 0.84)

### Data Quality
- **Missing Values**: Handled via imputation/placeholder strategy
- **Outliers**: Tree models naturally robust to outliers
- **Class Balance**: Not applicable (regression task)
- **Feature Scaling**: Not required for tree-based models

---

## Advanced Usage

### Retrain Models with New Data

```python
# In iphonetrain.py, update DATA_PATH
DATA_PATH = "./data/your_new_data.csv"

# Run training
python iphonetrain.py
```

### Modify Model Hyperparameters

Edit `train_models()` function in `iphonetrain.py`:

```python
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,        # Increase trees
        max_depth=20,            # Limit depth
        min_samples_split=5,     # Increase min samples
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    # ... other models
}
```

### Generate Predictions Programmatically

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("./models/random_forest.joblib")

# Prepare data (same format as app.py)
input_data = pd.DataFrame([{
    "Model": "iPhone 13 Pro",
    "storage": 256,
    "battery_health": 90,
    # ... other features
}])

# Predict
price = model.predict(input_data)[0]
print(f"Predicted price: {price:.0f} LKR")
```

---

## Troubleshooting

### Issue: `FileNotFoundError: data/iphone_release.csv`
**Solution**: Ensure CSV file is in the correct path relative to script location

### Issue: `ModuleNotFoundError: sklearn`
**Solution**: Run `pip install -r requirements.txt` in the virtual environment

### Issue: Streamlit app won't load
**Solution**: Run `streamlit run app.py --logger.level=debug` to see errors

### Issue: Model predictions don't match training results
**Solution**: Ensure input columns match training data format (use `prepare_input()` in app.py)

---

## Future Improvements

- [ ] Implement GridSearchCV for hyperparameter tuning
- [ ] Add cross-validation for more robust evaluation
- [ ] Include confidence intervals for predictions
- [ ] Support for other regression models (XGBoost, LightGBM)
- [ ] Feature engineering (interaction terms, polynomial features)
- [ ] Deploy app to cloud (Heroku, AWS, Azure)
- [ ] Add unit tests for data pipeline
- [ ] Create prediction history & analytics dashboard

---

## License

This project is provided as-is for educational and commercial use.

---

## Author & Contact

**Project**: iPhone Resale Price Prediction  
**Created**: 2024  
**Last Updated**: May 25, 2026

For questions or improvements, feel free to open an issue or contact the maintainers.

---

## Acknowledgments

- Dataset: iPhone resale market data from Sri Lanka
- ML Framework: scikit-learn
- UI Framework: Streamlit
- Python Community: For excellent libraries and documentation
