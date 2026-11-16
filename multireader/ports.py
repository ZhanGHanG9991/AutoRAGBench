"""Host-side service ports, kept outside the kernel's outgoing ephemeral range."""
import pathlib,socket

def candidates(job_id,ephemeral,span=60):
    low,high=ephemeral;start=int(job_id)%200
    return [base for k in range(200) for base in [10000+100*((start+k)%200)]
            if base+span-1<low or base>high]

def choose_port_base(job_id):
    low,high=map(int,pathlib.Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    for base in candidates(job_id,(low,high)):
        sockets=[]
        try:
            for port in range(base,base+60):
                s=socket.socket();sockets.append(s);s.bind(('127.0.0.1',port))
            return base
        except OSError:pass
        finally:
            for s in sockets:s.close()
    raise RuntimeError('No available service port block outside the ephemeral range')
