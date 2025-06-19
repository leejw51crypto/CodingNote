# AI Context & Tokenization Analysis Tools 🤖

A collection of educational tools demonstrating how AI models handle context, memory, and tokenization across different scenarios including auction analysis, memory persistence, and Unicode text processing.

## 🎯 Overview

This repository contains five Python scripts that explore different aspects of AI context processing and tokenization:

- **Auction Context Testing** - Tests AI's ability to analyze large datasets
- **Memory Persistence** - Demonstrates how AI maintains conversation context  
- **Unicode Tokenization** - Analyzes how different languages and scripts are tokenized
- **Wallet Address Processing** - Specialized tokenization for cryptocurrency addresses
- **Name Generation Testing** - Faker library integration for realistic test data

## 🚀 Features

### 🏠 Auction Analysis (`auction.py`)
- Generates realistic auction scenarios with 100+ bidders
- Tests AI's ability to identify highest bidders from large context
- Provides comprehensive accuracy analysis and confidence scoring
- Supports multiple test iterations with statistical reporting

### 🧠 Memory Demonstration (`memonto.py`)
- Shows how AI maintains persistent memory across conversations
- Includes general memory demo and specialized wallet address tracking
- Interactive chat mode with conversation history
- Streaming responses for real-time interaction

### 🌍 Unicode Tokenization (`unicode_tokenizer.py`)
- Analyzes tokenization across 14+ languages
- Tests emoji and special Unicode character processing
- Compares encoding efficiency across different scripts
- Interactive tokenizer for custom text analysis

### 💰 Wallet Tokenization (`wallet_tokenizer.py`)
- Specialized analysis for cryptocurrency wallet addresses
- Compares different address formats and storage methods
- Token cost analysis for efficient context usage
- Random address generation for testing

### 🎭 Name Generation (`test_faker_names.py`)
- Tests Faker library across multiple locales
- Demonstrates realistic name generation for different cultures
- Supports 8+ language locales

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-context-tokenization-tools.git
cd ai-context-tokenization-tools
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 🎮 Usage

### Auction Context Testing
```bash
python auction.py
```
Choose from:
- Multiple test iterations (recommended)
- Single test analysis
- Quick 3-iteration demo

### Memory Persistence Demo
```bash
python memonto.py
```
Options:
- General memory demonstration
- Wallet address context tracking
- Both demos sequentially

### Unicode Tokenization Analysis
```bash
python unicode_tokenizer.py
```
Includes:
- Multi-language analysis
- Emoji and special character testing
- Interactive tokenizer mode

### Wallet Address Tokenization
```bash
python wallet_tokenizer.py
```
Features:
- Address format comparison
- Token cost analysis
- Interactive address analyzer

### Name Generation Testing
```bash
python test_faker_names.py
```
Tests Faker name generation across different locales.

## 📊 Sample Output

### Auction Analysis Results
```
🎯 OVERALL PERFORMANCE:
   Total Tests: 10
   ✅ Complete Success: 8 (80.0%)
   ⚠️  Partial Success: 1 (10.0%)
   ❌ Failed: 1 (10.0%)
   📈 Average Confidence: 87.3%
```

### Tokenization Analysis
```
TOKENIZATION ANALYSIS: Chinese (Simplified)
======================================================================
Original text: 你好！你今天好吗？
Text length: 9 characters
🤖 Encoding: cl100k_base (GPT-4/GPT-3.5-turbo)
📊 Token count: 9
💰 Compression ratio: 1.0 chars per token
```

## 🔬 Educational Value

### Context Processing Insights
- **Large Context Handling**: Tests AI's ability to process 100+ data points
- **Memory Persistence**: Demonstrates conversation continuity
- **Accuracy Analysis**: Provides statistical validation of AI responses

### Tokenization Understanding
- **Cross-Language Efficiency**: Compare tokenization across scripts
- **Unicode Complexity**: Understand emoji and special character processing
- **Cost Optimization**: Analyze token usage for efficient prompt design

### Practical Applications
- **AI Testing Frameworks**: Template for systematic AI evaluation
- **Multilingual AI Development**: Tokenization insights for global applications
- **Context Optimization**: Strategies for efficient token usage

## 🛠️ Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls

## 📋 Dependencies

- `openai` - OpenAI API client
- `faker` - Realistic fake data generation
- `tiktoken` - OpenAI tokenization library

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Important Notes

- **API Costs**: These tools make OpenAI API calls. Monitor your usage to avoid unexpected charges.
- **Rate Limits**: Respect OpenAI's rate limits, especially when running multiple iterations.
- **Educational Purpose**: These tools are designed for learning and experimentation.

## 🌟 Use Cases

### For Developers
- Understanding AI context limitations
- Optimizing prompt design for token efficiency
- Testing multilingual AI applications

### For Researchers
- Systematic AI evaluation frameworks
- Cross-language tokenization studies
- Context processing benchmarking

### For Educators
- Demonstrating AI capabilities and limitations
- Teaching tokenization concepts
- Interactive AI learning tools

## 🔗 Related Resources

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Tiktoken Documentation](https://github.com/openai/tiktoken)
- [Faker Documentation](https://faker.readthedocs.io/)

## 📞 Support

If you find these tools helpful or have suggestions for improvements, please:
- ⭐ Star the repository
- 🐛 Report issues
- 💡 Submit feature requests
- 🤝 Contribute improvements

---

**Happy AI Experimenting! 🚀** 