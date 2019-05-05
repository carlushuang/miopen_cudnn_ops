from __future__ import print_function
import sys
import os
import subprocess

#CWD=os.path.dirname(os.path.realpath(__file__))
CWD=os.getcwd()

def run_cmd(cmd, cwd=CWD):
    proc = subprocess.Popen(cmd, cwd=cwd,
        stdout=sys.stdout, stderr=sys.stdout,shell=True)
    (out,_) = proc.communicate()

def get_pad(x):
    if x == 3:
        return 1
    if x == 5:
        return 2
    if x == 7:
        return 3
    return 0

def get_max_batch(w):
    if w>=192:
        return 16
    if w>=128:
        return 32
    if w>=96:
        return 64
    if w>=72:
        return 128
    if w>=55:
        return 256
    return 512

def need_break(w,c,k):
    return False

def need_continue(w,c,k):
    if k<c:
        return True
    return False

BUILD_DIR='./build'
range_x=[3,5,7]
range_w=[14,27,32,55,64,72,96,128,192]
#range_b=[16,32,64,128,256,512]
range_c=[64,128,192,256,384]
range_k=[128,256,384]

def run_op_driver(x,w,b,c,k,pad):
    conv_cmd=BUILD_DIR+'/op_driver conv -f 1 -k {} -w {} -h {} -c {} -x {} -s {} -p {} -n {}'\
        .format(k,w,w,c,x,1,pad,b)
    #print(conv_cmd)
    #print("---------------------------------------")
    run_cmd(conv_cmd)

print("N\tC\tH\tW\tK\tR\tS\tP\tQ\ttime(ms)\tmem\talgo")

for x in range_x:
    for w in range_w:
        b = get_max_batch(w)
        for c in range_c:
            for k in range_k:
                if need_break(w,c,k):
                    break
                if need_continue(w,c,k):
                    continue
                pad=get_pad(x)
                run_op_driver(x,w,b,c,k,pad)


