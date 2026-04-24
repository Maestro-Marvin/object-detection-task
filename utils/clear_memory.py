import gc
import torch
import psutil

def release_model(model) -> None:
    try:
        llm = getattr(model, "llm", None)
        if llm is not None:
            llm_engine = getattr(llm, "llm_engine", None)
            if llm_engine is not None and hasattr(llm_engine, "shutdown"):
                llm_engine.shutdown()
    except Exception:
        pass
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    
    current_pid = psutil.Process().pid
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            # FIX: Handle None cmdline safely
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            if 'VLLM::EngineCore' in cmdline or 'vllm' in cmdline.lower():
                proc.terminate()
                proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied):
            continue  # Also catch AccessDenied for robustness