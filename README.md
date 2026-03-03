# NovaTrade AI 🧠

NovaTrade AI is an advanced trading analytics dashboard built with Streamlit. It merges historical trader execution data with overall market sentiment (Fear & Greed Index) to identify profitable behaviors, cluster trading archetypes, and predict future account performance using machine learning.

## 🚀 Setup & Execution

### Prerequisites
Make sure you have Python installed. You must also place the two required data files into the project directory alongside `app.py`:
1. `historical_data.csv`
2. `fear_greed_index.csv`

### Installation
1. Clone or download this repository to your local machine.
2. Open a terminal and navigate to the project folder.
3. Install the required dependencies using the explicit requirements list:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
Run the Streamlit application from your terminal:
```bash
streamlit run app.py
```
The dashboard will automatically open in your default web browser (typically at `http://localhost:8501`).

---

## 📊 Short Write-Up: Analysis & Strategy

### Methodology
1. **Data Alignment & Engineering:** We aggregated individual trade executions to a daily level per account. We then merged these daily records with the crypto Fear & Greed Index to establish a direct link between market sentiment and individual trader behavior. We generated key metrics including daily PnL, Win Rate, Trade Frequency, Average Quote Size (Leverage Proxy), and lagged performance markers.
2. **Behavioral Archetyping (K-Means):** We utilized a K-Means clustering algorithm ($k=3$) to automatically segment traders based on their fundamental habits: Average Position Size, Trade Frequency, and Win Rate. 
3. **Predictive Modeling (Random Forest):** We trained a Random Forest Classifier to predict a binary target: *Will this trader be profitable tomorrow?* The model was trained dynamically using current sentiment, current PnL, trade volume, and crucially, 1-day lagged features for both sentiment and PnL.

### Key Insights
1. **Sentiment Drives Sizing:** There is a noticeable behavioral shift in risk appetite (proxied by `Size USD`) correlated tightly with extreme ends of the Fear & Greed Index. Traders actively modify their leverage in response to panic or euphoria.
2. **The "Frequent" Trap:** The clustering model clearly separates an archetype of 'High-Frequency' traders. Analysis reveals this segment often suffers from the highest cumulative fee drag and lowest normalized win rates when compared to 'Low-Volume/Swing' traders. 
3. **Momentum & Mean-Reversion in PnL:** The Random Forest Feature Importance analysis proves that `prev_day_pnl` is one of the strongest predictors of next-day profitability. Traders who suffered heavy losses the prior day are statistically prone to losing again, indicative of emotional "revenge trading" or forced liquidations.

### Strategy Recommendations
Based on the modeled outputs, we propose the following actionable rules of thumb:

*   **Rule 1: Sentiment-Based Scaling**  
    *During days classified as "Extreme Fear", algorithmically reduce maximum allowable position sizing by 30%. The models show that outsized losses cluster during these volatile periods. Protecting capital takes precedence over catching the exact bottom.*
*   **Rule 2: Implement "Cooling Off" Periods for High-Frequency Archetypes**  
    *If a trader's behavior maps to the "High-Frequency / Unprofitable" cluster, enforce a hard stop at a specific number of trades (e.g., 5 trades/day) during neutral or choppy market conditions. Only allow increased frequency during confirmed directional trends (Extreme Greed) where momentum strategies thrive.*
*   **Rule 3: The "Reset" Day**  
    *Because previous day PnL heavily influences future results, enforce a "Reset Rule": If `prev_day_pnl` falls below a critical negative threshold, the trader (or algorithm) must sit out the following 24 hours entirely or trade at 10% standard sizing to prevent emotional compounding of losses.*
