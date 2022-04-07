from pynvml import *


def nvidia_info():
    # pip install nvidia-ml-py
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    UNIT = 1024 * 1024
    try:
        nvmlInit()
        nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": nvmlDeviceGetName(handle),
                "total": memory_info.total / UNIT,
                "free": memory_info.free / UNIT,
                "used": memory_info.used / UNIT,
                "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict


def check_gpu_mem_used_rate(msg):
    info = nvidia_info()
    # print(info)
    gpu_cnt = len(info['gpus'])
    for i in range(gpu_cnt):
        gpu_name = info['gpus'][i]['gpu_name']
        used = info['gpus'][i]['used']
        tot = info['gpus'][i]['total']
        print(f"{msg}: gpu_{i} used: {used}MB, tot: {tot}MB, 使用率：{used / tot}")
