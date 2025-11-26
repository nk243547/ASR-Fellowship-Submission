'EOF'
# ASR Fellowship Challenge: Adapter-Based Fine-Tuning

## 📋 Submission Overview
Complete implementation of adapter-based fine-tuning for low-resource Kinyarwanda ASR using the provided WebDataset.

## 🚨 Important Note
The dataset (`DigitalUmuganda/ASR_Fellowship_Challenge_Dataset`) contains **only audio data without ground truth transcriptions**. Therefore, **WER cannot be calculated**.

## 🛠️ Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- Internet connection for dataset download

### Automated Pipeline (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/nk243547/ASR-Fellowship-Submission.git
cd ASR-Fellowship-Submission

# 2. Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run complete pipeline (download → train → evaluate)
./run.sh

