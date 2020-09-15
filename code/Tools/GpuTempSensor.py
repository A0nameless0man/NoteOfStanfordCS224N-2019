import os

def get_gpu_tem():
    # shell_str = "tem_line=`nvidia-smi | grep %` && tem1=`echo $tem_line | cut -d C -f 1` " \
    #             "&& tem2=`echo $tem1 | cut -d % -f 2` && echo $tem2"
    shell_str = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
    result = os.popen(shell_str)
    result_str = result.read()
    tem_str = result_str.split("\n")[0]
    result.close()
    return float(tem_str)