# Quick Start Guide

## Prerequisites
- Python 3.8+
- OpenAI API key

## Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here

# 3. Run the system
python main.py
```

## Usage

### Interactive Q&A
```bash
python main.py
```

### Rebuild Vector Database
```bash
python main.py setup
```

### View Demo
```bash
python demo.py
```

## Example Questions

- What are the symptoms of diabetes?
- How is hypertension treated?
- What's the difference between cold and flu?
- How can I prevent Type 2 diabetes?
- What medications treat high blood pressure?

## Adding Medical Documents

1. Add `.txt` files to `data/` directory
2. Run `python main.py setup`
3. Start asking questions!

## Project Structure

```
├── main.py           # Main application
├── demo.py          # Demonstration script
├── src/             # Source code modules
├── data/            # Medical documents
└── requirements.txt # Dependencies
```

## Troubleshooting

**"OPENAI_API_KEY not set"**
- Create `.env` file with your API key

**"No documents found"**
- Add `.txt` files to `data/` directory
- Run `python main.py setup`

**Import errors**
- Run `pip install -r requirements.txt`
