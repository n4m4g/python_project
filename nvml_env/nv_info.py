#!/usr/bin/env python3

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


def getBar(percentage, curr_max=None, unit=None):
    condition = 0 < percentage and percentage < 5
    num_bar = 1 if condition else percentage // 5
    if not curr_max:
        bar = "[{:20}{:>3}%]".format('='*(num_bar), percentage)
    else:
        bar = "[{:20}{:>3}%] {}{} / {}{}".format('='*(num_bar),
                                                 percentage,
                                                 curr_max[0],
                                                 unit,
                                                 curr_max[1],
                                                 unit)

    return bar


if __name__ == "__main__":
    nvmlInit()

    for i in range(nvmlDeviceGetCount()):

        handle = nvmlDeviceGetHandleByIndex(i)

        # device name
        device_name = nvmlDeviceGetName(handle).decode()
        print("[{}] {}".format(i, device_name))

        # driver version
        driver_version = nvmlSystemGetDriverVersion().decode()
        print("\tDriver Version: {}".format(driver_version))

        # utilize
        utilization = nvmlDeviceGetUtilizationRates(handle)
        # memory
        info = nvmlDeviceGetMemoryInfo(handle)
        mem_used = toMiB(info.used)
        # mem_free = toMiB(info.free)
        mem_total = toMiB(info.total)
        print("\tGPU Util: \t{}".format(getBar(utilization.gpu)))
        print("\tMEM Util: \t{}".format(getBar(utilization.memory,
                                               [mem_used, mem_total],
                                               'MiB')))

        # decoder usage
        utilization, samplingPeriodUs = nvmlDeviceGetDecoderUtilization(handle)
        print("\tDecoder Util: \t{}".format(getBar(utilization)))

        # fan speed
        fan_speed = nvmlDeviceGetFanSpeed(handle)
        print("\tFan Speed: \t{}".format(getBar(fan_speed)))

        # power state
        power_used = nvmlDeviceGetPowerUsage(handle)/1000
        power_limit = nvmlDeviceGetPowerManagementDefaultLimit(handle)/1000
        power_used = int(power_used)
        power_limit = int(power_limit)
        power_rate = int(power_used/power_limit)
        print("\tPower Util: \t{}".format(getBar(power_rate,
                                                 [power_used, power_limit],
                                                 'W')))

        # clock
        clk = nvmlDeviceGetClockInfo(handle, 0)
        max_clk = nvmlDeviceGetMaxClockInfo(handle, 0)
        print("\tGPU Clock: \t{}MHz / {}MHz".format(clk, max_clk))

        # temperature
        temp = nvmlDeviceGetTemperature(handle, 0)
        print("\tTemperature: \t{}C".format(temp))

        # processes
        compute_processes = nvmlDeviceGetComputeRunningProcesses(handle)
        graphic_processes = nvmlDeviceGetGraphicsRunningProcesses(handle)
        graphic_processes = sorted(graphic_processes,
                                   key=lambda x: toMiB(x.usedGpuMemory),
                                   reverse=True)
        print("\n\t=== Processes ===")
        for p in graphic_processes:
            p_pid = p.pid
            p_name = nvmlSystemGetProcessName(p_pid).decode()
            p_name = p_name.split(' ')[0].split('/')[-1]
            p_mem = toMiB(p.usedGpuMemory)
            print("\t{}: [{}MiB]".format(p_name, p_mem))

    nvmlShutdown()
