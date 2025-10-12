# 🚀 Alpha Agents - Fixed Version

A comprehensive stock analysis platform combining quantitative screening with AI-powered qualitative analysis.

## 📋 Overview

Alpha Agents provides two main functionalities:
1. **📊 Fundamental Screener** - Quantitative stock filtering based on financial metrics
2. **🤖 QualAgent Analysis** - AI-powered qualitative analysis using multiple LLM models

## 🛠️ Features

### Fundamental Screener
- ⚡ **Fast Mode**: Parallel processing for faster results
- 🔍 **Advanced Filtering**: By sector, industry, market cap
- 💾 **Database Storage**: Save results to PostgreSQL
- 📥 **Export Options**: CSV, Excel downloads
- 🎯 **Real-time Metrics**: Profit margins, ROE, ROA, debt ratios

### QualAgent Analysis
- 🤖 **Multi-LLM Analysis**: Uses multiple AI models for comprehensive analysis
- 📊 **Multiple Data Sources**: Screener results, database, CSV upload, manual input
- 📄 **Full Report Display**: View complete analysis reports
- 📥 **Flexible Downloads**: Individual reports, ZIP archives, JSON/CSV summaries
- 🔍 **Report Selection**: Choose which company results to view/download

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Database
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql://username:password@host:port/database
```

### 3. Run the Application
```bash
streamlit run alpha_agents_fixed.py
```

## 📊 How to Use

### Fundamental Screener

1. **Select Filters**:
   - Choose region (US)
   - Select filter type (Sector/Industry)
   - Set market cap range
   - Enable Fast Mode for faster results

2. **Run Screening**:
   - Click "🚀 Run Screen"
   - View results in the main area
   - Results are automatically saved for QualAgent

3. **Export/Save**:
   - Download CSV/Excel files
   - Save to PostgreSQL database
   - Get unique Run ID for later reference

### QualAgent Analysis

1. **Choose Data Source**:
   - **From Screener Results**: Use companies from current screening
   - **From PostgreSQL Database**: Load from previous runs
   - **Upload CSV**: Upload your own company list
   - **Manual Input**: Enter ticker symbols manually
   - **Sample Companies**: Use predefined examples

2. **Configure Analysis**:
   - Select analysis type (quick/comprehensive/expert-guided)
   - Set user ID
   - Estimate costs

3. **Run Analysis**:
   - Click "🔍 Run QualAgent Analysis"
   - Monitor progress
   - View results when complete

4. **View/Download Results**:
   - Select specific reports from dropdown
   - View full markdown reports
   - Download individual files or ZIP archives
   - Export results as JSON/CSV

## 🔧 Database Features

### Query Options
- **Company Selection**: Choose specific companies to analyze
- **Row Limits**: Limit number of companies loaded
- **Industry Filtering**: Filter by specific industries
- **Run ID Selection**: Choose from previous screening runs

### Data Storage
- **Automatic Saving**: Screener results saved to PostgreSQL
- **Run Tracking**: Each screening gets unique Run ID
- **Column Flexibility**: Handles different database schemas
- **Error Handling**: Graceful handling of missing columns

## 📁 File Structure

```
alpha-agents/
├── alpha_agents_fixed.py          # Main Streamlit application
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env                          # Database configuration
├── utils/
│   ├── stock_screener.py        # Original screener
│   ├── fast_stock_screener.py    # Optimized screener
│   └── db_util.py                # Database utilities
├── agents/
│   └── QualAgent/               # AI analysis system
└── pages/                        # Individual page modules
```

## 🎯 Key Improvements Made

### Performance Optimizations
- ⚡ **Fast Screener**: Parallel processing reduces screening time from 230s to ~30s
- 💾 **Caching**: Intelligent caching system for repeated queries
- 🔄 **Background Processing**: Non-blocking operations

### Database Integration
- 🗄️ **PostgreSQL Support**: Full database integration with fallback
- 🔍 **Smart Queries**: Flexible column handling and error recovery
- 📊 **Data Validation**: Automatic column existence checking

### User Experience
- 📄 **Full Report Display**: Complete analysis reports in Streamlit
- 🎯 **Report Selection**: Choose which reports to view/download
- 📥 **Multiple Download Formats**: ZIP, JSON, CSV, individual files
- 🔍 **Advanced Filtering**: Industry, company, and row limit options

### Error Handling
- 🛡️ **Encoding Fixes**: Resolved character encoding issues
- 🔧 **Column Validation**: Handles missing database columns gracefully
- ⚠️ **Graceful Failures**: Clear error messages and fallback options

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Check `.env` file has correct `DATABASE_URL`
   - Verify database server is running
   - Check network connectivity

2. **"Ticker not in index" Error**:
   - Fixed in current version
   - Database columns are automatically detected
   - Error handling shows available columns

3. **QualAgent Analysis Fails**:
   - Check company exists in QualAgent data
   - Verify API keys are configured
   - Check internet connectivity for LLM calls

4. **Slow Screening**:
   - Enable "⚡ Fast Mode" checkbox
   - Reduce max companies limit
   - Check internet connection for data fetching

### Debug Information
- Database columns are shown when loading data
- Error messages include helpful context
- Raw data displayed for troubleshooting

## 📈 Performance Metrics

- **Original Screener**: ~230 seconds for 100 companies
- **Fast Screener**: ~30 seconds for 100 companies
- **QualAgent Analysis**: 30s-10min depending on type
- **Database Operations**: <5 seconds for typical queries

## 🔮 Future Enhancements

- 📊 **Portfolio Management**: Track and manage investment portfolios
- 📈 **Performance Analytics**: Historical analysis and backtesting
- 🤖 **More LLM Models**: Additional AI models for analysis
- 📱 **Mobile Support**: Responsive design for mobile devices
- 🔔 **Alerts**: Real-time notifications for screening results

## 📞 Support

For issues or questions:
1. Check this README for common solutions
2. Review error messages in the application
3. Check database connectivity and configuration
4. Verify all dependencies are installed

---

**Version**: Fixed Edition  
**Last Updated**: October 2025  
**Status**: Production Ready ✅