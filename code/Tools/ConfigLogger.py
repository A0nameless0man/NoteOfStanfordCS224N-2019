import sys
import torch
import re


def LogConfig(verables, writer):
    writer.add_text("info/PyTorch version", torch.__version__)
    writer.add_text("info/Cuda version", str(torch.version.cuda))
    writer.add_text("info/Cuda Device Count", str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        writer.add_text(
            "info/Cuda device/ %d : %s" % (i, info.name),
            "CUDA CC %d.%d \nMEM %dMB \nMultiProcessorCount %d" %
            (info.major, info.minor, info.total_memory /
             (1024**2), info.multi_processor_count))
    constant_regex = re.compile("^([A-Z]+_)*[A-Z]+$")
    for (key, value) in verables.items():
        if constant_regex.match(key):
            writer.add_text("config/%s" % (key), str(value))
    writer.add_text("info/Python version", sys.version)