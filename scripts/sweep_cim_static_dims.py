#!/usr/bin/env python3
"""
Sweep X = 5 … 49, share one ATTEMPT-k folder, forward --jobs to children.

  python sweep_cim_static_dims.py --jobs 7
"""

import subprocess, shlex, sys, pathlib, datetime, re, argparse, textwrap

ROOT       = pathlib.Path(__file__).resolve().parent
OPT_SCRIPT = ROOT / "optimise_cim_static.py"

def make_sweep_root() -> pathlib.Path:
    today = datetime.date.today().strftime("%Y-%m-%d")
    base  = pathlib.Path("New_Optimisation") / today
    base.mkdir(parents=True, exist_ok=True)
    k=1
    while (d:=base/f"ATTEMPT-{k}").exists(): k+=1
    d.mkdir(); return d

RE_DONE = re.compile(r"^\[DONE] (.*)$", re.M)

def run_dim(dim:int, jobs:int, sweep_root:pathlib.Path):
    cmd = (f"{sys.executable} {OPT_SCRIPT} --dim {dim} "
           f"--jobs {jobs} --root {sweep_root}")
    res = subprocess.run(shlex.split(cmd),capture_output=True,text=True)
    print(res.stdout,end="");
    if res.stderr: print(res.stderr,file=sys.stderr,end="")
    folder="n/a"
    if (m:=RE_DONE.search(res.stdout)): folder=m.group(1).strip()
    status="OK" if res.returncode==0 else f"FAIL({res.returncode})"
    return dict(dim=dim,status=status,folder=folder)

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--jobs",type=int,default=1)
    args=pa.parse_args()

    sweep_root = make_sweep_root()
    rows=[run_dim(x,args.jobs,sweep_root) for x in range(5,50)]

    print("\nSummary")
    print(textwrap.dedent("""\
        dim | status | folder
        -----------------------------"""))
    for r in rows:
        print(f"{r['dim']:>3} | {r['status']:<6} | {r['folder']}")
