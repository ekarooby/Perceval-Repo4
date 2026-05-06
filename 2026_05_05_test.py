import perceval as pcvl
from perceval import RemoteProcessor

TOKEN = "your_token_here"
rp = RemoteProcessor("qpu:belenos", token=TOKEN)
jobs = rp.list_jobs()
print(f"Total: {len(jobs)}")
for j in jobs:
    print(f"  {j.id}  {j.status}")