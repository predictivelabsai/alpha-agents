"""
# Lohusalu Capital Management - Multi-Agent Equity Portfolio System

![Alpha Agents Banner](https://i.imgur.com/your-banner-image.png)  <!-- Replace with a real banner image -->

**Lohusalu Capital Management** is an advanced multi-agent system for equity portfolio construction, based on cutting-edge research in artificial intelligence and financial analysis. This Streamlit application demonstrates how specialized AI agents can collaborate, debate, and reach consensus to make sophisticated investment decisions.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-orange.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green.svg)](https://www.langchain.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18-blue.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

- **🤖 Multi-Agent Collaboration**: 5 specialized AI agents with domain expertise in fundamental analysis, sentiment analysis, valuation, business quality, and secular trends.
- **📈 Portfolio Construction**: Automated stock selection, risk-adjusted optimization, and diversification analysis.
- **📊 Advanced Analytics**: Interactive visualizations including heatmaps, performance charts, and portfolio optimization analysis.
- **🧪 Comprehensive Testing**: Extensive test suite with performance metrics, data generation, and validation of agent decision-making.
- **🌐 Web Interface**: User-friendly multi-page Streamlit application for easy interaction and analysis.

---

## 🏗️ System Architecture

The Lohusalu Capital Management system is built on a modern, modular architecture:

- **Multi-Agent Framework**: LangGraph-based workflow orchestration for structured debate and consensus.
- **AI Engine**: OpenAI GPT models for agent intelligence and reasoning.
- **Web Interface**: Streamlit for the interactive multi-page application.
- **Data Visualization**: Plotly for advanced, interactive charts and heatmaps.
- **Database**: SQLite for persistent storage of analyses and portfolios.

![System Architecture Diagram](https://i.imgur.com/your-architecture-diagram.png) <!-- Replace with a real diagram -->

---

## ⚙️ Installation

Follow these steps to set up and run the Alpha Agents system on your local machine.

### Prerequisites

- Python 3.10 or higher
- `pip` for package management
- `git` for cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/predictivelabsai/alpha-agents.git
cd alpha-agents
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your OpenAI API key.

```env
OPENAI_API_KEY="your-openai-api-key-here"
```

Replace `your-openai-api-key-here` with your actual OpenAI API key.

---

## 🚀 Running the Application

Once the installation is complete, you can run the Streamlit application.

### 1. Run the Test Suite (Optional but Recommended)

Before running the application, it is recommended to run the comprehensive test suite to ensure all components are working correctly and to generate initial analysis data.

```bash
python tests/run_all_tests.py
```

This will create a `test-data/` directory with CSV and JSON files containing agent analysis results.

### 2. Start the Streamlit Application

Run the following command to start the Streamlit server:

```bash
streamlit run Home.py
```

The application will be available at `http://localhost:8501` in your web browser.

---

## 📋 Usage Guide

The Alpha Agents application is organized into several pages, accessible from the sidebar navigation.

### 🏠 Home

The home page provides an overview of the system, its key features, and the specialized AI agents. It also shows a summary of system performance based on the latest test data.

### 📊 Stock Analysis

Analyze individual stocks by providing the stock symbol, company name, and other relevant details. The multi-agent system will perform a comprehensive analysis and provide recommendations, risk assessments, and detailed reasoning from each agent.

### 📊 Analytics

Explore advanced visualizations of the system's performance. This page includes:

- **Agent Consensus & Risk Heatmaps**: Visualize agent agreement and risk assessments across multiple stocks.
- **Performance Metrics**: View confidence score distributions and recommendation breakdowns.
- **Portfolio Optimization**: Analyze risk vs. expected return for a basket of stocks.
- **Sector Analysis**: Understand sector-wise performance and recommendations.

### 🎯 Portfolio Builder

Build and analyze diversified portfolios. Add multiple stocks to a portfolio and let the agents collaborate to provide investment decisions, sector allocation analysis, and overall portfolio recommendations.

### ℹ️ About

Learn more about the Alpha Agents system, its research foundation, system architecture, and technical implementation details.

---

## 🧪 Testing

The project includes a comprehensive testing framework to ensure reliability and validate agent performance.

- **Unit Tests**: Individual tests for each agent and core components.
- **Integration Tests**: Tests for the multi-agent system and its collaboration mechanisms.
- **Data Generation**: Scripts to generate test data for analysis and visualization.

To run all tests, use the following command:

```bash
python tests/run_all_tests.py
```

Test results and generated data are stored in the `test-data/` directory.

---

## 🌐 Deploying the NextJS Landing Page

The project includes a professional NextJS landing page for Lohusalu Capital Management located in the `web-ui/` directory. Here's how to deploy it to various platforms:

### 📁 Landing Page Structure

```
web-ui/
├── src/
│   └── app/
│       ├── page.tsx          # Main landing page
│       ├── layout.tsx        # App layout
│       └── globals.css       # Global styles
├── public/
│   ├── logo.png             # Company logo
│   ├── hero-skyline.jpg     # Hero background image
│   ├── buildings-perspective.jpg
│   └── modern-towers.jpg
├── package.json
└── next.config.ts
```

### 🚀 Deployment Options

#### Option 1: Vercel (Recommended)

Vercel is the easiest way to deploy NextJS applications:

1. **Fork/Clone the repository**:
   ```bash
   git clone https://github.com/predictivelabsai/alpha-agents.git
   cd alpha-agents/web-ui
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Test locally**:
   ```bash
   npm run dev
   ```

4. **Deploy to Vercel**:
   - Visit [vercel.com](https://vercel.com)
   - Connect your GitHub account
   - Import the repository
   - Set the **Root Directory** to `web-ui`
   - Deploy automatically

5. **Custom Domain** (Optional):
   - Add your custom domain in Vercel dashboard
   - Update DNS settings as instructed

#### Option 2: Netlify

1. **Build the application**:
   ```bash
   cd web-ui
   npm install
   npm run build
   ```

2. **Deploy to Netlify**:
   - Visit [netlify.com](https://netlify.com)
   - Drag and drop the `web-ui/out` folder (after running `npm run build`)
   - Or connect your GitHub repository
   - Set **Base directory** to `web-ui`
   - Set **Build command** to `npm run build`
   - Set **Publish directory** to `web-ui/out`

#### Option 3: Render.com

1. **Configure NextJS for static export** in `next.config.ts`:
   ```typescript
   /** @type {import('next').NextConfig} */
   const nextConfig = {
     output: 'export',
     trailingSlash: true,
     images: {
       unoptimized: true
     }
   }
   
   module.exports = nextConfig
   ```

2. **Create a new Static Site on Render**:
   - Visit [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Static Site"

3. **Configure build settings**:
   - **Root Directory**: `web-ui`
   - **Build Command**: `yarn; yarn build`
   - **Publish Directory**: `out`

**Note**: Render.com requires NextJS to be configured for static export to work as a Static Site. Alternatively, you can deploy as a Web Service for full NextJS features.

#### Option 4: GitHub Pages

1. **Enable static export** in `next.config.ts`:
   ```typescript
   /** @type {import('next').NextConfig} */
   const nextConfig = {
     output: 'export',
     trailingSlash: true,
     images: {
       unoptimized: true
     }
   }
   
   module.exports = nextConfig
   ```

2. **Build and export**:
   ```bash
   cd web-ui
   npm run build
   ```

3. **Deploy to GitHub Pages**:
   - Push the `out` folder to a `gh-pages` branch
   - Enable GitHub Pages in repository settings

### 🔧 Environment Configuration

The landing page includes:
- **Professional Design**: Based on Pershing Square Holdings structure
- **Responsive Layout**: Mobile and desktop optimized
- **High-Quality Images**: Professional skyscraper photography
- **Login Integration**: Direct link to the Streamlit application
- **SEO Optimized**: Meta tags and structured content

### 🎨 Customization

To customize the landing page:

1. **Update Company Information**:
   - Edit `src/app/page.tsx`
   - Modify contact details, team information, and performance metrics

2. **Replace Images**:
   - Add new images to `public/` directory
   - Update image references in `page.tsx`

3. **Styling Changes**:
   - Modify `src/app/globals.css`
   - Update Tailwind classes in components

4. **Logo Updates**:
   - Replace `public/logo.png` with your company logo
   - Ensure proper dimensions (recommended: 200x200px)

### 📊 Performance Optimization

The landing page is optimized for:
- **Fast Loading**: Optimized images and minimal JavaScript
- **SEO**: Proper meta tags and semantic HTML
- **Mobile First**: Responsive design for all devices
- **Accessibility**: WCAG compliant components

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request.

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature/your-feature-name`)
3. **Make your changes**
4. **Commit your changes** (`git commit -m 'Add some feature'`)
5. **Push to the branch** (`git push origin feature/your-feature-name`)
6. **Open a pull request**

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For any questions or inquiries, please contact the development team.

- **Project Lead**: [Your Name](mailto:your-email@example.com)
- **GitHub Repository**: [https://github.com/predictivelabsai/alpha-agents](https://github.com/predictivelabsai/alpha-agents)

"""

