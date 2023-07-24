"""
wrapper around run.c
mostly deals with the sentencepiece encoding/decoding
C code does all the transformer inference of the individual tokens
"""

from tokenizer import Tokenizer
import subprocess
import time

# specify your command
# command = ["./run", "out/model.bin"]
command = ["./target/release/llama2-rs", "out/model.bin"]

# Start the process
proc = subprocess.Popen(command, stdout=subprocess.PIPE)
enc = Tokenizer()

t0 = time.time()
tokens = []
last = ""
for line in proc.stdout:
    token = int(line.decode("utf-8").strip())
    dec = enc.decode(tokens + [token])
    chunk = dec[len(last) :]
    print(chunk, end="", flush=True)
    tokens.append(token)
    last = dec
t1 = time.time()
# seeking help: how can we do streaming inference in sentencepiece properly?
# or even delete sentencepiece entirely?

print(f"\nachieved tok/s: {len(tokens) / (t1 - t0)}")
proc.wait()
