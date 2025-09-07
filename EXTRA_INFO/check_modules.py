import importlib.util

modules_to_check = [
    ("cim_svp", "cim_svp"),
    ("cim_optimizer", "cim_optimizer"),
    ("fpylll", "fpylll"),
    ("optuna", "optuna"),
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "sklearn"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("networkx", "networkx"),
    ("tqdm", "tqdm"),
    ("typer", "typer"),
    ("yaml", "yaml"),
    ("ipykernel", "ipykernel"),
    ("jupyterlab", None),  # can't import directly
    ("pytest", "pytest"),
    ("black", "black"),
    ("isort", "isort"),
    ("mypy", "mypy"),
]

for name, mod_name in modules_to_check:
    if mod_name is None:
        print(f"[INFO] {name}: cannot check via import")
        continue
    if importlib.util.find_spec(mod_name) is not None:
        print(f"[OK] {name}")
    else:
        print(f"[MISSING] {name}")
