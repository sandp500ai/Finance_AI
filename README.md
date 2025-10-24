# 🚀 LangChain Finance AI Agent - AI-Powered Financial Analysis

A financial AI agent that provides institutional-grade investment insights using LangChain, OpenAI, and official SEC filings. Features quarterly-aligned stock analysis with SEC 10-Q earnings data for professional-level fundamental analysis.

## � Key Features
### 🧠 **AI-Powered Investment Insights**
- **Natural Language Analysis**: GPT-powered investment analysis using quarterly SEC fundamentals
- **Conversational Interface**: Interactive chat about quarterly earnings, trends, and performance
- **Market Research**: AI-powered sector analysis based on quarterly financial data
- **Investment Thesis**: Automated recommendations using SEC 10-Q quarterly fundamentals
- **Portfolio Analysis**: Multi-stock quarterly performance and correlation analysis
- **Smart Market Reports**: AI-generated reports focused on quarterly business performance

## 🚀 Quick Start
### Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/api-keys))
- Internet connection (for fetching financial data)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Finance_AI.git
   cd Finance_AI
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key:**
   ```bash
   # Option 1: Environment variable (recommended)
   export OPENAI_API_KEY='your-openai-api-key-here'
   
   # Option 2: Set in Python code
   # Pass api_key parameter when initializing the agent
   ```

4. **Run the AI agent:**
   ```bash
   python finance_ai_agent.py
   ```

##  Project Structure

```
Finance_AI/
│
├── finance_ai_agent.py              # Main AI agent with quarterly analysis
├── requirements.txt                # Dependencies (LangChain, yfinance, etc.)
├── README.md                       # This comprehensive guide
```

## 🧠 AI Capabilities

### LangChain Framework
- **ChatOpenAI Integration**: Uses GPT-4 or a new model for financial analysis
- **Prompt Templates**: Specialized prompts for financial analysis, market research, and investment recommendations
- **Sequential Chains**: Multi-step analysis workflows for comprehensive insights
- **Memory Management**: Maintains conversation context for interactive sessions
- **Tool Integration**: AI agents that can use multiple financial analysis tools

### AI Analysis Features
- **Investment Recommendations**: Clear Buy/Hold/Sell recommendations with confidence levels
- **Risk Assessment**: Comprehensive risk analysis with Low/Medium/High ratings
- **Competitive Analysis**: AI-powered competitive positioning and market dynamics
- **Growth Prospects**: Analysis of growth drivers, catalysts, and future outlook
- **Portfolio Optimization**: AI-driven diversification and allocation recommendations

## 🔧 Configuration

### OpenAI API Setup

The AI agent requires an OpenAI API key:

1. **Get API Key**: Sign up at [OpenAI Platform](https://platform.openai.com/)
2. **Set Environment Variable**: 
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
3. **Model Selection**: Choose between `gpt-3.5-turbo` (faster, cheaper) or `gpt-4` (more sophisticated)

## 📈 Data Sources

### 1. SEC EDGAR Database (10-Q Only)
- **Source**: https://data.sec.gov/api/xbrl/companyfacts
- **Data Type**: Official quarterly 10-Q filings only (excludes annual 10-K reports)
- **Metrics**: Quarterly earnings, revenue, assets, liabilities, shareholders' equity
- **AI Enhancement**: AI interprets quarterly SEC fundamentals for investment insights
- **Validation**: All data verified to be from 10-Q quarterly reports only

### 2. Yahoo Finance (Quarterly Aggregated)
- **Source**: Yahoo Finance via yfinance library
- **Data Type**: Stock prices aggregated by calendar quarters to match SEC periods
- **Metrics**: Quarterly open/close/high/low prices, volume, performance percentages
- **AI Enhancement**: AI analyzes quarterly price patterns and correlations with earnings
- **Alignment**: Perfect temporal alignment with SEC 10-Q reporting periods

### 3. OpenAI Models
- **Models**: GPT-3.5-turbo, GPT-4
- **Capabilities**: Natural language understanding, financial reasoning, market analysis
- **Applications**: Investment analysis, risk assessment, market research

## ⚠️ Important Notes

### AI Limitations
- **Market Conditions**: AI analysis is based on historical data and current information
- **Investment Decisions**: AI recommendations should be part of broader investment research
- **Model Updates**: AI capabilities improve with newer model versions
- **Data Dependency**: AI insights are only as good as the underlying financial data

### Cost Management
- **Monitor Usage**: Built-in cost tracking helps manage API expenses
- **Batch Operations**: Use bulk analysis features for cost efficiency
- **Model Selection**: Choose appropriate model based on analysis needs

### Data Quality
- **Real-time Data**: Stock prices and company information are updated regularly
- **AI Verification**: Cross-reference AI insights with traditional financial analysis
- **Error Handling**: Robust error handling for API failures and data issues

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: SandPAI
- **Email**: info@sandpai.io  
- **GitHub**: https://github.com/sanp500ai/Finance_AI

---

**🚨 Disclaimer**: This AI tool is for educational and research purposes only. AI-generated investment advice should not be used as the sole basis for investment decisions. The AI may make mistakes or have biases. Always consult with financial professionals, conduct your own research, and consider multiple sources before making investment decisions. Past performance does not guarantee future results.


