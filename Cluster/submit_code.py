import numpy as np
import itertools
import uuid
import paramiko

# Generate a random UUID (UUID4)
job_id = uuid.uuid4()


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


h_values = np.linspace(0, 5, 3)
phi_values = np.linspace(0.6, 1.0, 3)

repeats = 3

ssh_key_path = r"C:\Users\nolan\.ssh\id_ed25519"
k = paramiko.Ed25519Key.from_private_key_file(ssh_key_path)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("calcsub.curie.fr", username="nburban", pkey=k)

for params_dict in product_dict(**{"h": h_values, "phi": phi_values}):
    args = ",".join([f"{k}={v}" for k, v in params_dict.items()])
    cmd = rf"qsub -t 1-{repeats} -v {args},SUBMIT_JOB_ID={job_id} numerical_physics/Cluster/run_sim.pbs"
    # print(cmd)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)

ssh.close()
