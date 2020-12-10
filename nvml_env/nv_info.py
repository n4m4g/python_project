#!/usr/bin/env python3

from pynvml import nvmlInit
from pynvml import nvmlSystemGetDriverVersion
from pynvml import nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetName
from pynvml import nvmlDeviceGetUtilizationRates
from pynvml import nvmlDeviceGetMemoryInfo
from pynvml import nvmlDeviceGetTemperature
from pynvml import nvmlDeviceGetFanSpeed
from pynvml import nvmlDeviceGetPowerUsage
from pynvml import nvmlDeviceGetPowerManagementDefaultLimit
from pynvml import nvmlDeviceGetDecoderUtilization
from pynvml import nvmlShutdown


if __name__ == "__main__":
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    # device name, driver version
    driver_version = nvmlSystemGetDriverVersion().decode()
    device_name = nvmlDeviceGetName(handle).decode()
    print("GPU {}: {}, {}".format(0, device_name, driver_version))

    # utilize
    utilization = nvmlDeviceGetUtilizationRates(handle)
    print("GPU Util: \t{}%".format(utilization.gpu))
    print("MEM Util: \t{}%".format(utilization.memory))

    # memory
    info = nvmlDeviceGetMemoryInfo(handle)
    mem_total = int(info.total/1024/1024)
    mem_used = int(info.used/1024/1024)
    mem_free = int(info.free/1024/1024)
    print("Memory: \t{}MiB | {}MiB | {}MiB".format(mem_used,
                                                   mem_free,
                                                   mem_total))

    # temperature
    temp = nvmlDeviceGetTemperature(handle, 0)
    print("Temperature: \t{}C".format(temp))

    # fan speed
    fan_speed = nvmlDeviceGetFanSpeed(handle)
    print("Fan speed: \t{}%".format(fan_speed))

    # power state
    power_used = int(nvmlDeviceGetPowerUsage(handle)/1000)
    power_limit = int(nvmlDeviceGetPowerManagementDefaultLimit(handle)/1000)
    print("Power Util: \t{}W / {}W".format(power_used, power_limit))

    # decoder usage
    utilization, samplingPeriodUs = nvmlDeviceGetDecoderUtilization(handle)
    print("Decoder used: \t{}%".format(utilization))

    print()

    nvmlShutdown()
