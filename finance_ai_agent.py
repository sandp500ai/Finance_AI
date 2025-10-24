"""
LangChain Finance AI Agent - AI-Powered Financial Analysis

This AI agent uses LangChain and LLMs to provide intelligent financial analysis including:
- Natural language financial insights and recommendations
- AI-powered investment thesis generation
- Interactive chat interface for market discussions
- Intelligent risk assessment and portfolio optimization

"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# LangChain imports (updated for newer versions)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print("Warning: Some modern LangChain features may not be available.")

try:
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    try:
        from langchain.callbacks import get_openai_callback
    except ImportError:
        # Fallback for when callback is not available
        class MockCallback:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            total_tokens = 0
            total_cost = 0.0
        get_openai_callback = lambda: MockCallback()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    """Simplified data fetching for AI analysis"""
    
    def __init__(self):
        self.sec_base_url = "https://data.sec.gov"
        self.headers = {'User-Agent': 'Finance AI Agent (contact@example.com)'}
    
    def fetch_stock_data(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Fetch stock price data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty:
                hist = hist.reset_index()
                hist['ticker'] = ticker
            
            return hist
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_quarterly_stock_data(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Fetch stock price data aggregated by quarters to match SEC earnings periods"""
        try:
            print(f"Fetching quarterly stock data for {ticker}...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Reset index to work with dates
            hist = hist.reset_index()
            
            # Ensure Date column exists and is properly formatted
            if 'Date' not in hist.columns:
                raise ValueError("Date column not found in stock data")
            
            hist['Date'] = pd.to_datetime(hist['Date'])
            
            # Create quarterly periods (remove timezone info to avoid warnings)
            hist['Date'] = hist['Date'].dt.tz_localize(None) if hist['Date'].dt.tz is not None else hist['Date']
            hist['Quarter'] = hist['Date'].dt.to_period('Q')
            
            # Calculate quarterly aggregations
            quarterly_data = hist.groupby('Quarter').agg({
                'Open': 'first',      # Opening price of the quarter
                'Close': 'last',      # Closing price of the quarter
                'High': 'max',        # Highest price in the quarter
                'Low': 'min',         # Lowest price in the quarter
                'Volume': 'mean',     # Average daily volume
                'Date': ['first', 'last']  # Start and end dates of quarter
            }).round(2)
            
            # Flatten column names properly
            new_columns = []
            for col in quarterly_data.columns:
                if isinstance(col, tuple):
                    if col[1]:  # Multi-level column
                        new_columns.append(f"{col[0]}_{col[1]}")
                    else:  # Single level column
                        new_columns.append(col[0])
                else:
                    new_columns.append(str(col))
            
            quarterly_data.columns = new_columns
            quarterly_data = quarterly_data.rename(columns={
                'Date_first': 'quarter_start_date',
                'Date_last': 'quarter_end_date'
            })
            
            # Reset index to get Quarter as a column
            quarterly_data = quarterly_data.reset_index()
            
            # Convert Quarter period to end date for consistency with SEC data
            quarterly_data['quarter_end'] = quarterly_data['Quarter'].dt.end_time.dt.date
            quarterly_data['quarter_end'] = pd.to_datetime(quarterly_data['quarter_end'])
            
            # Calculate additional metrics using the flattened column names
            quarterly_data['price_change_percent'] = (
                (quarterly_data['Close_last'] - quarterly_data['Open_first']) / quarterly_data['Open_first'] * 100
            ).round(2)
            
            quarterly_data['price_range_percent'] = (
                (quarterly_data['High_max'] - quarterly_data['Low_min']) / quarterly_data['Low_min'] * 100
            ).round(2)
            
            # Calculate average price for the quarter
            quarterly_data['average_price'] = (
                (quarterly_data['Open_first'] + quarterly_data['Close_last'] + quarterly_data['High_max'] + quarterly_data['Low_min']) / 4
            ).round(2)
            
            # Rename columns for clarity
            quarterly_data = quarterly_data.rename(columns={
                'Open_first': 'Open',
                'Close_last': 'Close', 
                'High_max': 'High',
                'Low_min': 'Low',
                'Volume_mean': 'Volume'
            })
            
            # Add ticker information
            quarterly_data['ticker'] = ticker
            
            # Reorder columns for clarity
            column_order = [
                'ticker', 'Quarter', 'quarter_end', 'quarter_start_date', 'quarter_end_date',
                'Open', 'Close', 'High', 'Low', 'average_price', 
                'price_change_percent', 'price_range_percent', 'Volume'
            ]
            
            quarterly_data = quarterly_data[column_order]
            
            print(f"Generated {len(quarterly_data)} quarters of stock price data")
            return quarterly_data
            
        except Exception as e:
            logger.error(f"Error fetching quarterly stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_company_info(self, ticker: str) -> Dict:
        """Get company information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'company_name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'current_price': info.get('currentPrice'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield')
            }
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def get_cik_from_ticker(self, ticker: str) -> str:
        """Get CIK (Central Index Key) from ticker symbol"""
        try:
            # SEC company tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                companies = response.json()
                
                # Search for the ticker
                for company_data in companies.values():
                    if company_data.get('ticker', '').upper() == ticker.upper():
                        cik = str(company_data.get('cik_str', '')).zfill(10)  # Pad with zeros
                        return cik
                        
            logger.warning(f"CIK not found for ticker {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
            return None
    
    def fetch_sec_earnings_data(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Fetch comprehensive quarterly financial data from SEC 10-Q filings only"""
        try:
            print(f"Fetching SEC quarterly financial data (10-Q only) for {ticker}...")
            
            # Get CIK first
            cik = self.get_cik_from_ticker(ticker)
            if not cik:
                logger.error(f"Could not find CIK for {ticker}")
                return pd.DataFrame()
            
            # Get company facts
            url = f"{self.sec_base_url}/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"SEC API request failed with status {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Look for various financial metrics in the facts
            financial_data = []
            
            # Financial metrics to extract
            metrics_map = {
                # Net Income
                'earnings': ['NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic', 'IncomeLossFromContinuingOperations'],
                # Revenue
                'revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomer'],
                # Total Assets
                'total_assets': ['Assets', 'AssetsCurrent'],
                # Total Liabilities  
                'total_liabilities': ['Liabilities', 'LiabilitiesCurrent'],
                # Shareholders Equity
                'shareholders_equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest']
            }
            
            facts = data.get('facts', {})
            us_gaap = facts.get('us-gaap', {})
            
            # Create a dictionary to store data by date
            data_by_date = {}
            
            # Extract data for each metric
            for metric_name, tags in metrics_map.items():
                for tag in tags:
                    if tag in us_gaap:
                        metric_facts = us_gaap[tag]
                        units = metric_facts.get('units', {})
                        
                        # Look for USD values
                        if 'USD' in units:
                            for entry in units['USD']:
                                # Only get quarterly data from 10-Q filings (exclude 10-K annual reports)
                                if entry.get('form') == '10-Q':
                                    date_key = entry.get('end')
                                    
                                    if date_key not in data_by_date:
                                        data_by_date[date_key] = {
                                            'date': date_key,
                                            'form': entry.get('form'),
                                            'filed': entry.get('filed')
                                        }
                                    
                                    # Convert to billions and store
                                    data_by_date[date_key][metric_name] = entry.get('val', 0) / 1e9
                            break  # Use first available tag for this metric
            
            if not data_by_date:
                logger.warning(f"No quarterly financial data found in SEC 10-Q filings for {ticker}")
                return pd.DataFrame()
            
            # Convert to list of dictionaries
            financial_data = list(data_by_date.values())
            
            # Create DataFrame
            df = pd.DataFrame(financial_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter by years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            df = df[df['date'] >= start_date]
            
            # Remove duplicates, keeping the most recent filing for each quarter
            df = df.drop_duplicates(subset=['date'], keep='last')
            
            # Fill missing values with NaN for cleaner analysis
            for col in ['earnings', 'revenue', 'total_assets', 'total_liabilities', 'shareholders_equity']:
                if col not in df.columns:
                    df[col] = pd.NA
            
            print(f"Found {len(df)} quarters of SEC 10-Q data with metrics: {[col for col in df.columns if col not in ['date', 'form', 'filed']]}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching SEC quarterly financial data (10-Q) for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_sec_company_concepts(self, ticker: str) -> Dict:
        """Fetch detailed SEC company concepts and metadata"""
        try:
            cik = self.get_cik_from_ticker(ticker)
            if not cik:
                return {}
            
            url = f"{self.sec_base_url}/api/xbrl/companyconcepts/CIK{cik}.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Could not fetch SEC company concepts for {ticker}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching SEC company concepts for {ticker}: {e}")
            return {}

class AIFinancialAnalysisTool:
    """Financial data analysis tool (simplified)"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def _run(self, ticker: str, years: int = 5) -> str:
        """Execute financial analysis for a given ticker"""
        try:
            # Get stock data (both daily and quarterly)
            stock_data = self.data_fetcher.fetch_stock_data(ticker, years)
            quarterly_stock_data = self.data_fetcher.fetch_quarterly_stock_data(ticker, years)
            company_info = self.data_fetcher.get_company_info(ticker)
            
            # Get SEC earnings data
            sec_earnings_data = self.data_fetcher.fetch_sec_earnings_data(ticker, years)
            
            # Calculate basic metrics
            analysis = {
                'ticker': ticker,
                'company_info': company_info,
                'price_analysis': {},
                'quarterly_price_analysis': {},
                'financial_metrics': {},
                'sec_earnings_analysis': {}
            }
            
            # Daily price analysis
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                initial_price = stock_data['Close'].iloc[0]
                price_change = ((current_price - initial_price) / initial_price) * 100
                
                analysis['price_analysis'] = {
                    'current_price': current_price,
                    'price_change_percent': price_change,
                    'max_price': stock_data['High'].max(),
                    'min_price': stock_data['Low'].min(),
                    'avg_volume': stock_data['Volume'].mean(),
                    'volatility': stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
                }
            
            # Quarterly price analysis (aligned with SEC reporting)
            if not quarterly_stock_data.empty:
                recent_quarters = quarterly_stock_data.tail(8)  # Last 8 quarters
                
                analysis['quarterly_price_analysis'] = {
                    'quarters_available': len(quarterly_stock_data),
                    'most_recent_quarter_price': recent_quarters['Close'].iloc[-1] if len(recent_quarters) > 0 else None,
                    'average_quarter_price': recent_quarters['average_price'].mean(),
                    'quarterly_volatility': recent_quarters['price_change_percent'].std(),
                    'best_quarter_performance': recent_quarters['price_change_percent'].max(),
                    'worst_quarter_performance': recent_quarters['price_change_percent'].min(),
                    'quarters_with_gains': len(recent_quarters[recent_quarters['price_change_percent'] > 0]),
                    'recent_quarters_data': [
                        {
                            'quarter': str(row['Quarter']),
                            'quarter_end_date': row['quarter_end'].strftime('%Y-%m-%d'),
                            'opening_price': row['Open'],
                            'closing_price': row['Close'],
                            'average_price': row['average_price'],
                            'price_change_percent': row['price_change_percent'],
                            'price_range_percent': row['price_range_percent'],
                            'average_volume': row['Volume']
                        }
                        for _, row in recent_quarters.iterrows()
                    ]
                }
            
            # Add company metrics
            analysis['financial_metrics'] = {
                'pe_ratio': company_info.get('pe_ratio'),
                'market_cap': company_info.get('market_cap'),
                'beta': company_info.get('beta'),
                'dividend_yield': company_info.get('dividend_yield')
            }
            
            # Add comprehensive SEC financial analysis
            if not sec_earnings_data.empty:
                recent_data = sec_earnings_data.tail(8)  # Last 8 quarters
                
                # Calculate financial ratios and trends
                sec_analysis = {
                    'quarters_available': len(sec_earnings_data),
                    'latest_filing_date': recent_data['date'].iloc[-1].strftime('%Y-%m-%d') if len(recent_data) > 0 else None,
                    'earnings_analysis': {},
                    'revenue_analysis': {},
                    'balance_sheet_analysis': {},
                    'financial_ratios': {},
                    'recent_quarters_data': []
                }
                
                # Earnings analysis
                if 'earnings' in recent_data.columns and recent_data['earnings'].notna().any():
                    earnings_data = recent_data['earnings'].dropna()
                    if len(earnings_data) > 0:
                        sec_analysis['earnings_analysis'] = {
                            'most_recent_earnings_billions': round(earnings_data.iloc[-1], 2),
                            'average_quarterly_earnings_billions': round(earnings_data.mean(), 2),
                            'earnings_trend': 'improving' if len(earnings_data) >= 2 and earnings_data.iloc[-1] > earnings_data.iloc[-2] else 'declining',
                            'earnings_volatility': round(earnings_data.std(), 2),
                            'earnings_growth_yoy': round(((earnings_data.iloc[-1] / earnings_data.iloc[-4] - 1) * 100), 2) if len(earnings_data) >= 4 else None
                        }
                
                # Revenue analysis
                if 'revenue' in recent_data.columns and recent_data['revenue'].notna().any():
                    revenue_data = recent_data['revenue'].dropna()
                    if len(revenue_data) > 0:
                        sec_analysis['revenue_analysis'] = {
                            'most_recent_revenue_billions': round(revenue_data.iloc[-1], 2),
                            'average_quarterly_revenue_billions': round(revenue_data.mean(), 2),
                            'revenue_trend': 'growing' if len(revenue_data) >= 2 and revenue_data.iloc[-1] > revenue_data.iloc[-2] else 'declining',
                            'revenue_growth_yoy': round(((revenue_data.iloc[-1] / revenue_data.iloc[-4] - 1) * 100), 2) if len(revenue_data) >= 4 else None
                        }
                
                # Balance sheet analysis
                balance_metrics = ['total_assets', 'total_liabilities', 'shareholders_equity']
                for metric in balance_metrics:
                    if metric in recent_data.columns and recent_data[metric].notna().any():
                        metric_data = recent_data[metric].dropna()
                        if len(metric_data) > 0:
                            sec_analysis['balance_sheet_analysis'][f'{metric}_billions'] = round(metric_data.iloc[-1], 2)
                
                # Financial ratios
                if ('total_assets' in recent_data.columns and 'total_liabilities' in recent_data.columns and 
                    recent_data['total_assets'].notna().any() and recent_data['total_liabilities'].notna().any()):
                    assets = recent_data['total_assets'].dropna().iloc[-1] if recent_data['total_assets'].notna().any() else None
                    liabilities = recent_data['total_liabilities'].dropna().iloc[-1] if recent_data['total_liabilities'].notna().any() else None
                    
                    if assets and liabilities:
                        sec_analysis['financial_ratios']['debt_to_assets_ratio'] = round(liabilities / assets, 3)
                        sec_analysis['financial_ratios']['equity_ratio'] = round((assets - liabilities) / assets, 3)
                
                # Recent quarters summary
                for _, row in recent_data.iterrows():
                    quarter_data = {
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'form_type': row['form'],
                        'filed_date': row['filed']
                    }
                    
                    # Add available financial metrics
                    for metric in ['earnings', 'revenue', 'total_assets', 'total_liabilities', 'shareholders_equity']:
                        if metric in row and pd.notna(row[metric]):
                            quarter_data[f'{metric}_billions'] = round(row[metric], 2)
                    
                    sec_analysis['recent_quarters_data'].append(quarter_data)
                
                analysis['sec_financial_analysis'] = sec_analysis
            else:
                analysis['sec_financial_analysis'] = {
                    'status': 'No SEC quarterly financial data available',
                    'note': 'This may be due to company not filing 10-Q reports with SEC, being a private company, or quarterly data not yet available'
                }
            
            # Add correlation analysis between quarterly stock performance and SEC earnings
            if not quarterly_stock_data.empty and not sec_earnings_data.empty:
                correlation_analysis = self._analyze_stock_earnings_correlation(quarterly_stock_data, sec_earnings_data)
                analysis['stock_earnings_correlation'] = correlation_analysis
            else:
                analysis['stock_earnings_correlation'] = {
                    'status': 'Insufficient data for correlation analysis'
                }
            
            return json.dumps(analysis, indent=2, default=str)
            
        except Exception as e:
            return f"Error analyzing {ticker}: {str(e)}"
    
    def _analyze_stock_earnings_correlation(self, quarterly_stock_data: pd.DataFrame, sec_earnings_data: pd.DataFrame) -> Dict:
        """Analyze correlation between quarterly stock performance and SEC earnings"""
        try:
            # Merge quarterly stock data with SEC earnings data by quarter end dates
            # Convert SEC earnings date to quarter end for matching
            sec_data = sec_earnings_data.copy()
            sec_data['quarter_end'] = pd.to_datetime(sec_data['date'])
            
            # Match quarters (allow some flexibility in date matching)
            merged_data = []
            
            for _, stock_row in quarterly_stock_data.iterrows():
                stock_quarter_end = stock_row['quarter_end']
                
                # Find SEC data within 45 days of quarter end (to account for filing delays)
                date_range_start = stock_quarter_end - timedelta(days=45)
                date_range_end = stock_quarter_end + timedelta(days=45)
                
                matching_sec = sec_data[
                    (sec_data['quarter_end'] >= date_range_start) & 
                    (sec_data['quarter_end'] <= date_range_end)
                ]
                
                if not matching_sec.empty:
                    # Use the closest match
                    closest_match = matching_sec.iloc[0]
                    
                    merged_row = {
                        'quarter': str(stock_row['Quarter']),
                        'quarter_end': stock_quarter_end.strftime('%Y-%m-%d'),
                        'stock_price_change_percent': stock_row['price_change_percent'],
                        'average_quarter_price': stock_row['average_price'],
                        'sec_earnings_billions': closest_match.get('earnings', None),
                        'sec_revenue_billions': closest_match.get('revenue', None),
                        'sec_filing_date': closest_match['filed'] if 'filed' in closest_match else None
                    }
                    merged_data.append(merged_row)
            
            if not merged_data:
                return {'status': 'No matching quarterly data found'}
            
            merged_df = pd.DataFrame(merged_data)
            
            # Calculate correlations
            correlations = {}
            insights = []
            
            # Stock price vs earnings correlation
            if merged_df['sec_earnings_billions'].notna().sum() >= 3:  # Need at least 3 data points
                earnings_data = merged_df[merged_df['sec_earnings_billions'].notna()]
                price_earnings_corr = earnings_data['stock_price_change_percent'].corr(earnings_data['sec_earnings_billions'])
                correlations['price_vs_earnings'] = round(price_earnings_corr, 3)
                
                if abs(price_earnings_corr) > 0.5:
                    direction = "positive" if price_earnings_corr > 0 else "negative"
                    strength = "strong" if abs(price_earnings_corr) > 0.7 else "moderate"
                    insights.append(f"Stock price shows {strength} {direction} correlation with earnings ({price_earnings_corr:.2f})")
            
            # Stock price vs revenue correlation (if available)
            if merged_df['sec_revenue_billions'].notna().sum() >= 3:
                revenue_data = merged_df[merged_df['sec_revenue_billions'].notna()]
                price_revenue_corr = revenue_data['stock_price_change_percent'].corr(revenue_data['sec_revenue_billions'])
                correlations['price_vs_revenue'] = round(price_revenue_corr, 3)
                
                if abs(price_revenue_corr) > 0.5:
                    direction = "positive" if price_revenue_corr > 0 else "negative"
                    strength = "strong" if abs(price_revenue_corr) > 0.7 else "moderate"
                    insights.append(f"Stock price shows {strength} {direction} correlation with revenue ({price_revenue_corr:.2f})")
            
            # Performance analysis
            positive_earnings_quarters = merged_df[merged_df['sec_earnings_billions'] > 0]
            if not positive_earnings_quarters.empty:
                avg_price_performance_positive_earnings = positive_earnings_quarters['stock_price_change_percent'].mean()
                insights.append(f"Average stock performance in profitable quarters: {avg_price_performance_positive_earnings:.1f}%")
            
            return {
                'matched_quarters': len(merged_data),
                'correlations': correlations,
                'insights': insights,
                'quarterly_data': merged_data[-8:] if len(merged_data) > 8 else merged_data,  # Last 8 quarters
                'summary': f"Analyzed {len(merged_data)} quarters with matching stock and SEC data"
            }
            
        except Exception as e:
            return {'error': f"Correlation analysis failed: {str(e)}"}

class LangChainFinanceAI:
    """
    AI-Powered Finance Agent using LangChain and LLMs
    """
    
    def __init__(self, openai_api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """Initialize the AI Finance Agent"""
        
        # Set up API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        self.model_name = model_name
        self.data_fetcher = DataFetcher()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.1)
        
        # Initialize tools
        self.financial_tool = AIFinancialAnalysisTool()
        
        # Create analysis prompts
        self._setup_analysis_prompts()
        
        # Simple conversation history
        self.conversation_history = []
    
    def _setup_analysis_prompts(self):
        """Set up AI analysis prompts"""
        
        # Main financial analysis prompt
        self.analysis_prompt = PromptTemplate(
            input_variables=["financial_data", "ticker"],
            template="""
You are an expert financial analyst. Analyze the following data for {ticker}:

{financial_data}

Provide a comprehensive analysis including:

1. **SEC Quarterly Earnings & Fundamental Analysis**
   - Analyze the SEC quarterly earnings data from 10-Q filings only
   - Quarterly earnings trends and consistency across reporting periods
   - Revenue and profitability trajectory on a quarterly basis
   - Compare quarterly SEC filing data with market expectations

2. **Quarterly Stock Performance vs. SEC Data**
   - Analyze quarterly stock performance aligned with SEC reporting periods
   - Stock price correlation with earnings announcements
   - Market reaction to quarterly SEC filings
   - Price-to-earnings relationship over quarterly periods

3. **Financial Health Assessment**
   - Current valuation metrics (P/E, market cap analysis)
   - Quarterly price performance and volatility analysis
   - Liquidity and trading volume assessment
   - Integration of SEC filing insights with quarterly market data

3. **Investment Analysis**
   - Strengths and competitive advantages based on SEC data
   - Key risks and challenges revealed in filings
   - Growth prospects and catalysts from earnings trends
   - Quality of earnings assessment

4. **Market Position**
   - Sector comparison and industry trends
   - Competitive positioning based on financial performance
   - Market sentiment vs. fundamental SEC data

5. **Investment Recommendation**
   - Clear Buy/Hold/Sell recommendation based on SEC fundamentals
   - Target price range (if applicable)
   - Investment timeframe
   - Risk rating (Low/Medium/High)
   - Confidence level (1-10)

6. **Key Monitoring Points**
   - Important SEC filing dates and metrics to watch
   - Upcoming earnings announcements and catalysts
   - Risk factors from both market and SEC filing perspectives

Focus heavily on the SEC earnings data as the foundation for fundamental analysis. Be specific, data-driven, and provide actionable insights for investors.
"""
        )
        
        # Market research prompt
        self.market_research_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are a market research expert. Provide analysis on: {query}

Include:
1. Current market conditions and trends
2. Sector-specific insights
3. Economic factors and their impact
4. Competitive landscape analysis
5. Future outlook and key themes

Provide actionable insights for investment decisions.
"""
        )
        
        # Portfolio analysis prompt
        self.portfolio_prompt = PromptTemplate(
            input_variables=["portfolio_data"],
            template="""
You are a portfolio manager. Analyze this portfolio:

{portfolio_data}

Provide:
1. **Portfolio Overview**
   - Diversification assessment
   - Sector/geographic allocation
   - Risk-return profile

2. **Performance Analysis**
   - Top performers and laggards
   - Risk-adjusted returns
   - Correlation analysis

3. **Recommendations**
   - Rebalancing suggestions
   - Position sizing adjustments
   - Risk management improvements
   - New investment opportunities

4. **Portfolio Grade**
   - Overall grade (A-F)
   - Rationale for the grade
   - Key improvement areas

Be specific with actionable recommendations.
"""
        )
    
    def _invoke_llm_with_prompt(self, prompt: PromptTemplate, **kwargs) -> str:
        """Helper method to invoke LLM with a prompt template"""
        try:
            formatted_prompt = prompt.format(**kwargs)
            response = self.llm.invoke(formatted_prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _market_research_tool(self, query: str) -> str:
        """Market research tool"""
        try:
            response = self._invoke_llm_with_prompt(self.market_research_prompt, query=query)
            return response
        except Exception as e:
            return f"Market research error: {str(e)}"
    
    def _portfolio_analysis_tool(self, tickers: str) -> str:
        """Portfolio analysis tool"""
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            portfolio_data = {}
            
            for ticker in ticker_list[:10]:  # Limit to 10 stocks
                try:
                    data = self.financial_tool._run(ticker, years=3)
                    portfolio_data[ticker] = json.loads(data)
                except Exception as e:
                    portfolio_data[ticker] = {"error": str(e)}
            
            response = self._invoke_llm_with_prompt(
                self.portfolio_prompt,
                portfolio_data=json.dumps(portfolio_data, indent=2, default=str)
            )
            return response
            
        except Exception as e:
            return f"Portfolio analysis error: {str(e)}"
    
    def analyze_stock(self, ticker: str, years: int = 5) -> Dict[str, Any]:
        """
        Comprehensive AI stock analysis
        
        Args:
            ticker (str): Stock ticker symbol
            years (int): Years of historical data
            
        Returns:
            Dict: Complete analysis results with AI insights
        """
        logger.info(f"Starting AI analysis for {ticker}")
        
        try:
            with get_openai_callback() as cb:
                # Get financial data
                financial_data = self.financial_tool._run(ticker, years)
                
                # Run AI analysis
                ai_analysis = self._invoke_llm_with_prompt(
                    self.analysis_prompt,
                    financial_data=financial_data,
                    ticker=ticker
                )
                
                # Compile results
                results = {
                    "ticker": ticker,
                    "analysis_date": datetime.now().isoformat(),
                    "raw_data": json.loads(financial_data),
                    "ai_analysis": ai_analysis,
                    "ai_usage": {
                        "tokens_used": cb.total_tokens,
                        "estimated_cost": cb.total_cost,
                        "model": self.model_name
                    }
                }
                
                # Save results
                self._save_results(results, ticker)
                
                logger.info(f"Analysis complete. Cost: ${cb.total_cost:.4f}, Tokens: {cb.total_tokens}")
                return results
                
        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def chat_about_stock(self, ticker: str, question: str) -> str:
        """Ask questions about a specific stock"""
        try:
            # Get current data
            financial_data = self.financial_tool._run(ticker, years=3)
            
            context_prompt = f"""
            Based on this financial data for {ticker}:
            {financial_data}
            
            Question: {question}
            
            Provide a helpful, accurate response based on the data and financial expertise.
            """
            
            response = self.llm.invoke(context_prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
            
        except Exception as e:
            return f"Error discussing {ticker}: {str(e)}"
    
    def generate_market_report(self, tickers: List[str], focus: str = "general") -> str:
        """Generate market analysis report"""
        try:
            # Analyze multiple stocks
            analyses = {}
            for ticker in tickers[:8]:  # Limit for cost control
                try:
                    data = self.financial_tool._run(ticker, years=2)
                    analyses[ticker] = json.loads(data)
                except Exception as e:
                    analyses[ticker] = {"error": str(e)}
            
            # Generate report
            report_prompt = f"""
            Create a market analysis report focusing on: {focus}
            
            Stock Analysis Data:
            {json.dumps(analyses, indent=2, default=str)}
            
            Structure the report as:
            1. Executive Summary
            2. Market Overview & Key Themes
            3. Individual Stock Highlights
            4. Sector Analysis & Trends
            5. Risk Assessment
            6. Investment Recommendations
            7. Outlook & Key Catalysts
            
            Make it professional and actionable for investors.
            """
            
            report = self.llm.invoke(report_prompt)
            
            # Handle different response types
            if hasattr(report, 'content'):
                report_content = report.content
            else:
                report_content = str(report)
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"market_report_{focus}_{timestamp}.md"
            
            os.makedirs("reports", exist_ok=True)
            with open(f"reports/{filename}", 'w') as f:
                f.write(f"# Market Analysis Report - {focus.title()}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report_content)
            
            return report_content
            
        except Exception as e:
            return f"Report generation failed: {str(e)}"
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("🤖 Finance AI Agent - Interactive Chat")
        print("=" * 50)
        print("Ask me about stocks, markets, or investment strategies!")
        print("Commands:")
        print("  /analyze TICKER - Full analysis of a stock")
        print("  /portfolio TICKER1,TICKER2,... - Portfolio analysis")
        print("  /report TICKER1,TICKER2,... - Generate market report")
        print("  /quit - Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['/quit', 'quit', 'exit']:
                    print("👋 Goodbye! Happy investing!")
                    break
                
                elif user_input.startswith('/analyze '):
                    ticker = user_input[9:].strip().upper()
                    print(f"\n🔍 Analyzing {ticker}...")
                    
                    results = self.analyze_stock(ticker)
                    if 'error' not in results:
                        print(f"\n📊 AI Analysis for {ticker}:")
                        print("=" * 40)
                        print(results['ai_analysis'])
                        print(f"\nCost: ${results['ai_usage']['estimated_cost']:.4f}")
                    else:
                        print(f"❌ Error: {results['error']}")
                
                elif user_input.startswith('/portfolio '):
                    tickers = user_input[11:].strip()
                    print(f"\n📈 Analyzing portfolio: {tickers}")
                    
                    analysis = self._portfolio_analysis_tool(tickers)
                    print(f"\n📊 Portfolio Analysis:")
                    print("=" * 40)
                    print(analysis[:1500] + "..." if len(analysis) > 1500 else analysis)
                
                elif user_input.startswith('/report '):
                    tickers = user_input[8:].strip().split(',')
                    tickers = [t.strip().upper() for t in tickers]
                    print(f"\n📋 Generating market report for: {', '.join(tickers)}")
                    
                    report = self.generate_market_report(tickers)
                    print(f"\n📄 Market Report Preview:")
                    print("=" * 40)
                    print(report[:2000] + "..." if len(report) > 2000 else report)
                    print("\n📁 Full report saved to 'reports' folder")
                
                else:
                    # General conversation
                    print(f"\n🤖 AI: ", end="")
                    try:
                        # Simple chat implementation
                        self.conversation_history.append(f"User: {user_input}")
                        
                        # Create context from recent conversation
                        context = "\n".join(self.conversation_history[-5:])  # Last 5 exchanges
                        
                        chat_prompt = f"""
You are a financial AI assistant. Based on this conversation context:

{context}

User's latest message: {user_input}

Provide a helpful response about finance, investments, or markets. If the user asks about specific stocks, suggest using the /analyze command for detailed analysis.
"""
                        
                        response = self.llm.invoke(chat_prompt)
                        
                        # Handle different response types
                        if hasattr(response, 'content'):
                            ai_response = response.content
                        else:
                            ai_response = str(response)
                        
                        print(ai_response)
                        self.conversation_history.append(f"AI: {ai_response}")
                        
                    except Exception as e:
                        print(f"I encountered an error: {e}")
                        
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _save_results(self, results: Dict, ticker: str):
        """Save analysis results"""
        try:
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/{ticker}_analysis_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def create_visualization(self, ticker: str, years: int = 5):
        """Create quarterly stock price visualization with SEC 10-Q earnings data"""
        try:
            # Get both daily and quarterly stock data
            daily_stock_data = self.data_fetcher.fetch_stock_data(ticker, years)
            quarterly_stock_data = self.data_fetcher.fetch_quarterly_stock_data(ticker, years)
            
            if daily_stock_data.empty and quarterly_stock_data.empty:
                print(f"No stock data available for {ticker}")
                return
            
            # Get SEC financial data
            sec_data = self.data_fetcher.fetch_sec_earnings_data(ticker, years)
            
            # Use a more compatible style
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    plt.style.use('default')
            
            # Create figure with dual y-axes
            fig, ax1 = plt.subplots(figsize=(16, 10))
            
            # Plot stock price on left y-axis
            color1 = '#2E86AB'
            color1_light = '#87CEEB'
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Stock Price ($)', color=color1, fontsize=12)
            
            # Plot daily stock price as a thin line (if available)
            if not daily_stock_data.empty:
                ax1.plot(daily_stock_data['Date'], daily_stock_data['Close'], 
                        linewidth=1, color=color1_light, alpha=0.6, label='Daily Stock Price')
            
            # Plot quarterly average stock prices as bold line with markers
            if not quarterly_stock_data.empty:
                ax1.plot(quarterly_stock_data['quarter_end'], quarterly_stock_data['average_price'], 
                        'o-', linewidth=3, markersize=8, color=color1, 
                        label='Quarterly Avg Price', alpha=0.9)
                
                # Add quarterly price change annotations
                for _, row in quarterly_stock_data.tail(8).iterrows():
                    change_color = 'green' if row['price_change_percent'] > 0 else 'red'
                    ax1.annotate(f'{row["price_change_percent"]:.1f}%', 
                               (row['quarter_end'], row['average_price']),
                               textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=8,
                               color=change_color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for earnings
            ax2 = ax1.twinx()
            color2 = '#FF6B6B'
            ax2.set_ylabel('Quarterly Net Income ($ Billions)', color=color2, fontsize=12)
            
            # Plot SEC earnings data if available
            if not sec_data.empty:
                # Ensure all dates are properly formatted as datetime
                sec_data_copy = sec_data.copy()  # Avoid modifying original
                sec_data_copy['date'] = pd.to_datetime(sec_data_copy['date'])
                
                # Get date range from available stock data
                if not quarterly_stock_data.empty:
                    start_date = quarterly_stock_data['quarter_end'].iloc[0]
                    end_date = quarterly_stock_data['quarter_end'].iloc[-1]
                elif not daily_stock_data.empty:
                    daily_stock_data['Date'] = pd.to_datetime(daily_stock_data['Date'])
                    start_date = daily_stock_data['Date'].iloc[0]
                    end_date = daily_stock_data['Date'].iloc[-1]
                else:
                    start_date = sec_data_copy['date'].min()
                    end_date = sec_data_copy['date'].max()
                
                # Use query method to avoid datetime comparison issues
                sec_filtered = sec_data_copy.query('date >= @start_date and date <= @end_date')
                
                if not sec_filtered.empty and 'earnings' in sec_filtered.columns:
                    # Filter out NaN earnings values
                    earnings_filtered = sec_filtered[sec_filtered['earnings'].notna()]
                    
                    if not earnings_filtered.empty:
                        # Plot as scatter points with connecting lines
                        line2 = ax2.plot(earnings_filtered['date'], earnings_filtered['earnings'], 
                                        'o-', linewidth=2, markersize=8, color=color2, 
                                        label='SEC Net Income', alpha=0.8)
                        
                        # Add value labels on points
                        for _, row in earnings_filtered.iterrows():
                            ax2.annotate(f'${row["earnings"]:.1f}B', 
                                       (row['date'], row['earnings']),
                                       textcoords="offset points", 
                                       xytext=(0,10), ha='center', fontsize=9,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                            
                        print(f"Plotted {len(earnings_filtered)} quarters of SEC earnings data")
                    else:
                        print("No valid SEC earnings data in the selected timeframe")
                        ax2.text(0.5, 0.5, 'No Valid SEC Earnings Data\nfor Selected Period', 
                                transform=ax2.transAxes, ha='center', va='center',
                                fontsize=12, color=color2, alpha=0.7)
                else:
                    print("No SEC earnings data available")
                    ax2.text(0.5, 0.5, 'No SEC Earnings Data Available', 
                            transform=ax2.transAxes, ha='center', va='center',
                            fontsize=12, color=color2, alpha=0.7)
            else:
                ax2.text(0.5, 0.5, 'No SEC Financial Data Available', 
                        transform=ax2.transAxes, ha='center', va='center',
                        fontsize=12, color=color2, alpha=0.7)
            
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Add title
            ax1.set_title(f'{ticker} Quarterly Stock Performance and SEC Earnings - Past {years} Years', 
                         fontsize=16, fontweight='bold', pad=20)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format x-axis
            fig.autofmt_xdate()
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs("plots", exist_ok=True)
            filename = f"plots/{ticker}_quarterly_price_sec_earnings_chart.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Quarterly chart with SEC earnings data saved to {filename}")
            
            # Print some statistics
            print(f"\n📊 Chart Statistics:")
            if not daily_stock_data.empty:
                print(f"Daily Price Range: ${daily_stock_data['Close'].min():.2f} - ${daily_stock_data['Close'].max():.2f}")
            if not quarterly_stock_data.empty:
                print(f"Quarterly Avg Price Range: ${quarterly_stock_data['average_price'].min():.2f} - ${quarterly_stock_data['average_price'].max():.2f}")
                print(f"Best Quarter Performance: {quarterly_stock_data['price_change_percent'].max():.1f}%")
                print(f"Worst Quarter Performance: {quarterly_stock_data['price_change_percent'].min():.1f}%")
            
            if not sec_data.empty:
                # Use date range from quarterly data if available
                if not quarterly_stock_data.empty:
                    stock_start = quarterly_stock_data['quarter_end'].iloc[0]
                    stock_end = quarterly_stock_data['quarter_end'].iloc[-1]
                elif not daily_stock_data.empty:
                    stock_start = daily_stock_data['Date'].iloc[0] 
                    stock_end = daily_stock_data['Date'].iloc[-1]
                else:
                    stock_start = sec_data['date'].min()
                    stock_end = sec_data['date'].max()
                
                sec_filtered_stats = sec_data.query('date >= @stock_start and date <= @stock_end')
                if not sec_filtered_stats.empty and 'earnings' in sec_filtered_stats.columns:
                    earnings_stats = sec_filtered_stats[sec_filtered_stats['earnings'].notna()]
                    if not earnings_stats.empty:
                        print(f"SEC Earnings Range: ${earnings_stats['earnings'].min():.1f}B - ${earnings_stats['earnings'].max():.1f}B")
                        print(f"Average Quarterly Net Income: ${earnings_stats['earnings'].mean():.1f}B")
                        print(f"Number of Quarters (SEC data): {len(earnings_stats)}")
                        
                        # Show filing information
                        print(f"\nSEC Filing Details:")
                        for _, row in earnings_stats.iterrows():
                            earnings_val = f"${row['earnings']:.1f}B" if pd.notna(row['earnings']) else "N/A"
                            print(f"  {row['date'].strftime('%Y-%m-%d')}: {earnings_val} ({row['form']} filed {row['filed']})")
            
        except Exception as e:
            print(f"Visualization error: {e}")

def main():
    """Main function"""
    print("🚀 LangChain Finance AI Agent")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found in environment!")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key required to continue.")
            return
    
    try:
        # Initialize AI agent
        print("\n🤖 Initializing AI agent...")
        ai = LangChainFinanceAI(openai_api_key=api_key)
        
        print("✅ AI agent ready!")
        print("\nOptions:")
        print("1. Analyze single stock")
        print("2. Interactive chat")  
        print("3. Portfolio analysis")
        print("4. Market report")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            ticker = input("Enter ticker: ").upper().strip()
            years = int(input("Years of data (default 5): ") or 5)
            
            print(f"\n🔍 Analyzing {ticker}...")
            results = ai.analyze_stock(ticker, years)
            
            if 'error' not in results:
                print(f"\n📊 AI Analysis for {ticker}:")
                print("=" * 50)
                print(results['ai_analysis'])
                print(f"\n💰 Cost: ${results['ai_usage']['estimated_cost']:.4f}")
                
                # Optional: Create chart
                create_chart = input("\nCreate price chart? (y/n): ").lower()
                if create_chart == 'y':
                    ai.create_visualization(ticker, years)
            else:
                print(f"❌ Error: {results['error']}")
                
        elif choice == "2":
            ai.interactive_chat()
            
        elif choice == "3":
            tickers = input("Enter tickers (comma-separated): ").strip()
            analysis = ai._portfolio_analysis_tool(tickers)
            print(f"\n📊 Portfolio Analysis:")
            print("=" * 50)
            print(analysis)
            
        elif choice == "4":
            tickers_input = input("Enter tickers for report: ").strip()
            tickers = [t.strip().upper() for t in tickers_input.split(',')]
            focus = input("Report focus (e.g., 'tech sector', 'dividend stocks'): ") or "general"
            
            print(f"\n📋 Generating market report...")
            report = ai.generate_market_report(tickers, focus)
            print(f"\n📄 Market Report:")
            print("=" * 50)
            print(report[:2000] + "..." if len(report) > 2000 else report)
            
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()