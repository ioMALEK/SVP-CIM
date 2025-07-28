# SVP-CIM Project

This repository implements an approximate Shortest Vector Problem (SVP) solver using a Coherent Ising Machine (CIM) simulator.

## Structure

- `svp_cim.py`: Main Python script implementing the SVP-CIM heuristic.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Common ignores for Python and Xcode.
- `README.md`: Project overview and setup instructions.

## Setup (macOS Silicon)

1. Clone this repository.
2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Open the project in Xcode:
   - Open Xcode and choose "Open another project..."
   - Navigate to this directory and select it to create an Xcode project workspace.
   - You can add `svp_cim.py` to an Xcode script target for easier execution.

## Usage

Run the solver script:
```bash
python svp_cim.py
```

Adjust hyperparameters in `svp_cim.py` as needed.
