#!/usr/bin/env python3

from collections import defaultdict

from pynvml import nvmlInit
from pynvml import nvmlSystemGetDriverVersion
from pynvml import nvmlDeviceGetCount
from pynvml import nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetName
from pynvml import nvmlDeviceGetClockInfo
from pynvml import nvmlDeviceGetMaxClockInfo
from pynvml import nvmlDeviceGetUtilizationRates
from pynvml import nvmlDeviceGetMemoryInfo
from pynvml import nvmlDeviceGetTemperature
from pynvml import nvmlDeviceGetFanSpeed
from pynvml import nvmlDeviceGetPowerUsage
from pynvml import nvmlDeviceGetPowerManagementDefaultLimit
from pynvml import nvmlDeviceGetDecoderUtilization
from pynvml import nvmlDeviceGetComputeRunningProcesses
from pynvml import nvmlDeviceGetGraphicsRunningProcesses
from pynvml import nvmlSystemGetProcessName
from pynvml import nvmlShutdown


def toMiB(value):
    return int(value/1024/1024)


def getBar(percentage, msg=None):
    condition = 0 < percentage and percentage < 10
    num_bar = 1 if condition else percentage // 10
    if not msg:
        bar = "[{:10}{:>3}%]".format('='*(num_bar), percentage)
    else:
        bar = "[{:10}{:>3}%] {}".format('='*(num_bar),
                                        percentage, msg)
    return bar


def pack_msg(values, unit):
    if isinstance(values, list):
        assert len(values) == 2
        return f"{values[0]}{unit} / {values[1]}{unit}"
    elif isinstance(values, str):
        return values


def show_process(header, processes):
    print(header)
    processes = sorted(processes,
                       key=lambda x: toMiB(x.usedGpuMemory),
                       reverse=True)
    buf = defaultdict()
    for p in processes:
        p_pid = p.pid
        p_name = nvmlSystemGetProcessName(p_pid).decode()
        p_name = p_name.split(' ')[0].split('/')[-1]
        p_mem = toMiB(p.usedGpuMemory)
        buf[p_name] = p_mem

    message = [f"{k}: [{v}MiB]" for k, v in buf.items()]
    print('\n'.join(message))
    # print("{}: [{}MiB]".format(p_name, p_mem))


if __name__ == "__main__":
    nvmlInit()

    for i in range(nvmlDeviceGetCount()):
        info = defaultdict()

        driver_version = nvmlSystemGetDriverVersion().decode()
        info['Driver Version'] = driver_version

        info['GPU idx'] = i
        handle = nvmlDeviceGetHandleByIndex(i)

        # device name
        device_name = nvmlDeviceGetName(handle).decode()
        info['GPU Name'] = device_name

        # clock
        clk = nvmlDeviceGetClockInfo(handle, 0)
        max_clk = nvmlDeviceGetMaxClockInfo(handle, 0)
        clk_rate = int(clk/max_clk*100)
        msg = pack_msg([clk, max_clk], 'MHz')
        info['GPU Clock'] = getBar(clk_rate, msg)

        # utilize
        util = nvmlDeviceGetUtilizationRates(handle)
        # memory
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        mem_used = toMiB(mem_info.used)
        # mem_free = toMiB(info.free)
        mem_total = toMiB(mem_info.total)
        info['GPU Util'] = getBar(util.gpu)
        mem_rate = int(mem_used/mem_total*100)
        msg = pack_msg([mem_used, mem_total], 'MiB')
        info['MEM Util'] = getBar(mem_rate, msg)

        # decoder usage
        utilization, samplingPeriodUs = nvmlDeviceGetDecoderUtilization(handle)
        info['DEC Util'] = getBar(utilization)

        # power state
        power_used = nvmlDeviceGetPowerUsage(handle)/1000
        power_limit = nvmlDeviceGetPowerManagementDefaultLimit(handle)/1000
        power_used = int(power_used)
        power_limit = int(power_limit)
        power_rate = int(power_used/power_limit*100)
        msg = pack_msg([power_used, power_limit], 'W')
        info['Power Util'] = getBar(power_rate, msg)

        # fan speed, temperature
        fan_speed = nvmlDeviceGetFanSpeed(handle)
        temp = nvmlDeviceGetTemperature(handle, 0)
        msg = f"{temp}C"
        info['Fan Speed'] = getBar(fan_speed, msg)

        message = [f"{k} \t{v}" for k, v in info.items()]
        print('\n'.join(message))

        # graphic processes
        graphic_processes = nvmlDeviceGetGraphicsRunningProcesses(handle)
        header = "\n=== Graphic Processes ==="
        show_process(header, graphic_processes)

        # graphic processes
        compute_processes = nvmlDeviceGetComputeRunningProcesses(handle)
        header = "\n=== Compute Processes ==="
        show_process(header, compute_processes)

    nvmlShutdown()
