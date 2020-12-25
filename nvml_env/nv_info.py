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


if __name__ == "__main__":
    nvmlInit()

    for i in range(nvmlDeviceGetCount()):

        handle = nvmlDeviceGetHandleByIndex(i)

        # device name
        device_name = nvmlDeviceGetName(handle).decode()
        print("GPU {}: {}".format(i, device_name))

        # driver version
        driver_version = nvmlSystemGetDriverVersion().decode()
        print("Driver Version: {}".format(driver_version))

        # clock
        clk = nvmlDeviceGetClockInfo(handle, 0)
        max_clk = nvmlDeviceGetMaxClockInfo(handle, 0)
        print("GPU clk: \t{}MHz / {}MHz".format(clk, max_clk))

        # utilize
        utilization = nvmlDeviceGetUtilizationRates(handle)
        print("GPU Util: \t{}%".format(utilization.gpu))
        print("MEM Util: \t{}%".format(utilization.memory))

        # memory
        info = nvmlDeviceGetMemoryInfo(handle)
        mem_used = toMiB(info.used)
        mem_free = toMiB(info.free)
        mem_total = toMiB(info.total)
        print("Memory: \t{}MiB / {}MiB / {}MiB".format(mem_used,
                                                       mem_free,
                                                       mem_total))

        # temperature
        temp = nvmlDeviceGetTemperature(handle, 0)
        print("Temperature: \t{}C".format(temp))

        # fan speed
        fan_speed = nvmlDeviceGetFanSpeed(handle)
        print("Fan speed: \t{}%".format(fan_speed))

        # power state
        power_used = nvmlDeviceGetPowerUsage(handle)/1000
        power_limit = nvmlDeviceGetPowerManagementDefaultLimit(handle)/1000
        print("Power Util: \t{}W / {}W".format(int(power_used),
                                               int(power_limit)))

        # decoder usage
        utilization, samplingPeriodUs = nvmlDeviceGetDecoderUtilization(handle)
        print("Decoder used: \t{}%".format(utilization))

        # processes
        compute_processes = nvmlDeviceGetComputeRunningProcesses(handle)
        graphic_processes = nvmlDeviceGetGraphicsRunningProcesses(handle)
        graphic_processes = sorted(graphic_processes,
                                   key=lambda x: toMiB(x.usedGpuMemory),
                                   reverse=True)
        print("\n=== Processes ===")
        for p in graphic_processes:
            p_pid = p.pid
            p_name = nvmlSystemGetProcessName(p_pid).decode()
            p_name = p_name.split(' ')[0].split('/')[-1]
            p_mem = toMiB(p.usedGpuMemory)
            print("{}: [{}MiB]".format(p_name, p_mem))

    nvmlShutdown()
