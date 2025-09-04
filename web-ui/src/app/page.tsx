import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Image
                src="/logo.png"
                alt="Lohusalu Capital"
                width={40}
                height={40}
                className="mr-3"
              />
              <span className="text-xl font-bold text-gray-900">Lohusalu Capital Management</span>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-8">
                <a href="#about" className="text-gray-700 hover:text-gray-900 px-3 py-2 text-sm font-medium">About</a>
                <a href="#strategy" className="text-gray-700 hover:text-gray-900 px-3 py-2 text-sm font-medium">Investment Strategy</a>
                <a href="#performance" className="text-gray-700 hover:text-gray-900 px-3 py-2 text-sm font-medium">Performance</a>
                <a href="#team" className="text-gray-700 hover:text-gray-900 px-3 py-2 text-sm font-medium">Team</a>
                <a href="#contact" className="text-gray-700 hover:text-gray-900 px-3 py-2 text-sm font-medium">Contact</a>
                <Link 
                  href="https://lohusalucap.predictivelabs.ai/"
                  className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
                >
                  Login
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center">
        <div className="absolute inset-0">
          <Image
            src="/hero-skyline.jpg"
            alt="Financial District"
            fill
            className="object-cover"
            priority
          />
          <div className="absolute inset-0 bg-black bg-opacity-50"></div>
        </div>
        <div className="relative z-10 text-center text-white max-w-4xl mx-auto px-4">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
            Lohusalu Capital Management
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-gray-200">
            Leveraging advanced AI and multi-agent systems for superior equity portfolio construction
          </p>
          <div className="space-x-4">
            <Link 
              href="https://lohusalucap.predictivelabs.ai/"
              className="bg-blue-600 text-white px-8 py-3 rounded-md text-lg font-medium hover:bg-blue-700 transition-colors inline-block"
            >
              Access Platform
            </Link>
            <a 
              href="#about"
              className="border border-white text-white px-8 py-3 rounded-md text-lg font-medium hover:bg-white hover:text-gray-900 transition-colors inline-block"
            >
              Learn More
            </a>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">About Lohusalu Capital Management</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              We are a forward-thinking investment management firm that combines traditional financial expertise 
              with cutting-edge artificial intelligence to deliver exceptional results for our clients.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Data-Driven Insights</h3>
              <p className="text-gray-600">
                Our AI-powered analysis processes vast amounts of financial data to identify opportunities 
                that traditional methods might miss.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Multi-Agent Technology</h3>
              <p className="text-gray-600">
                Five specialized AI agents collaborate to analyze fundamentals, sentiment, valuation, 
                business quality, and technology trends.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Risk Management</h3>
              <p className="text-gray-600">
                Sophisticated risk assessment and portfolio optimization ensure your investments 
                are protected while maximizing returns.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Investment Strategy Section */}
      <section id="strategy" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl font-bold text-gray-900 mb-6">Our Investment Strategy</h2>
              <p className="text-lg text-gray-600 mb-6">
                Lohusalu Capital Management employs a revolutionary multi-agent approach to equity analysis, 
                combining the expertise of five specialized AI agents to provide comprehensive 
                investment recommendations.
              </p>
              <div className="space-y-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mr-3 mt-1">
                    <span className="text-white text-sm font-bold">1</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Fundamental Analysis</h4>
                    <p className="text-gray-600">Deep dive into financial statements and company metrics</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mr-3 mt-1">
                    <span className="text-white text-sm font-bold">2</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Sentiment Analysis</h4>
                    <p className="text-gray-600">Real-time market sentiment and news analysis</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mr-3 mt-1">
                    <span className="text-white text-sm font-bold">3</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Technical Valuation</h4>
                    <p className="text-gray-600">Price momentum and technical indicator analysis</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mr-3 mt-1">
                    <span className="text-white text-sm font-bold">4</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Business Quality Assessment</h4>
                    <p className="text-gray-600">7-step framework for evaluating business excellence</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mr-3 mt-1">
                    <span className="text-white text-sm font-bold">5</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Technology Trends</h4>
                    <p className="text-gray-600">Secular trend analysis and future positioning</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="relative">
              <Image
                src="/buildings-perspective.jpg"
                alt="Modern Architecture"
                width={600}
                height={400}
                className="rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Performance Section */}
      <section id="performance" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Performance Metrics</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our AI-driven approach has consistently delivered superior risk-adjusted returns 
              across various market conditions.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600 mb-2">15.2%</div>
              <div className="text-gray-600">Annual Return</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600 mb-2">1.8</div>
              <div className="text-gray-600">Sharpe Ratio</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600 mb-2">12.5%</div>
              <div className="text-gray-600">Max Drawdown</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600 mb-2">85%</div>
              <div className="text-gray-600">Win Rate</div>
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section id="team" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Leadership Team</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our team combines decades of investment experience with cutting-edge AI expertise.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-32 h-32 bg-gray-300 rounded-full mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Dr. Sarah Chen</h3>
              <p className="text-blue-600 mb-2">Chief Investment Officer</p>
              <p className="text-gray-600 text-sm">
                Former Goldman Sachs MD with 15 years in quantitative strategies
              </p>
            </div>
            <div className="text-center">
              <div className="w-32 h-32 bg-gray-300 rounded-full mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Michael Rodriguez</h3>
              <p className="text-blue-600 mb-2">Head of AI Research</p>
              <p className="text-gray-600 text-sm">
                PhD in Machine Learning, former Google DeepMind researcher
              </p>
            </div>
            <div className="text-center">
              <div className="w-32 h-32 bg-gray-300 rounded-full mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">James Thompson</h3>
              <p className="text-blue-600 mb-2">Chief Executive Officer</p>
              <p className="text-gray-600 text-sm">
                20+ years in asset management, former BlackRock portfolio manager
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <div>
              <h2 className="text-4xl font-bold mb-6">Get in Touch</h2>
              <p className="text-xl text-gray-300 mb-8">
                Ready to experience the future of investment management? 
                Contact us to learn more about our AI-powered approach.
              </p>
              <div className="space-y-4">
                <div className="flex items-center">
                  <svg className="w-6 h-6 text-blue-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span>New York, NY 10001</span>
                </div>
                <div className="flex items-center">
                  <svg className="w-6 h-6 text-blue-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                  <span>+1 (555) 123-4567</span>
                </div>
                <div className="flex items-center">
                  <svg className="w-6 h-6 text-blue-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <span>contact@lohusalucapital.com</span>
                </div>
              </div>
            </div>
            <div className="relative">
              <Image
                src="/modern-towers.jpg"
                alt="Modern Financial District"
                width={600}
                height={400}
                className="rounded-lg"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center mb-4">
                <Image
                  src="/logo.png"
                  alt="Lohusalu Capital"
                  width={32}
                  height={32}
                  className="mr-2"
                />
                <span className="text-lg font-bold">Lohusalu Capital Management</span>
              </div>
              <p className="text-gray-400 text-sm">
                Intelligent capital management powered by advanced AI technology.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#about" className="hover:text-white">About</a></li>
                <li><a href="#strategy" className="hover:text-white">Strategy</a></li>
                <li><a href="#team" className="hover:text-white">Team</a></li>
                <li><a href="#contact" className="hover:text-white">Contact</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Services</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white">Portfolio Management</a></li>
                <li><a href="#" className="hover:text-white">AI Analytics</a></li>
                <li><a href="#" className="hover:text-white">Risk Assessment</a></li>
                <li><a href="#" className="hover:text-white">Research</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white">Terms of Service</a></li>
                <li><a href="#" className="hover:text-white">Disclosures</a></li>
                <li><a href="#" className="hover:text-white">Compliance</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>&copy; 2025 Lohusalu Capital Management. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
