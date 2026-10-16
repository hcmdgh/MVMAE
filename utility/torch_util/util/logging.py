from ..common import * 

ERROR = 1
WARNING = 2
INFO = 3
DEBUG = 4 

terminal_ptr = sys.stderr 
file_ptr = None  
start_time = time.time() 
verbose = DEBUG 


def get_now_datetime_str() -> str:
    now = datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')

    
def set_verbose(_verbose: int):
    global verbose 
    verbose = _verbose 


def set_terminal(enable: bool = True):
    global terminal_ptr 
    
    if enable: 
        terminal_ptr = sys.stderr 
    else:
        terminal_ptr = None 


def set_log_file(path: Optional[str] = './log.log'):
    global file_ptr 
    
    if file_ptr:
        file_ptr.close() 
        
    if path:
        file_ptr = open(path, 'w', encoding='utf-8')
    else:
        file_ptr = None 
        
        
def disable_logging():
    set_verbose(0)
        
    
def log(message: Any,
        level: int):
    duration = int(time.time() - start_time)
    duration = "%02d:%02d" % (duration // 60, duration % 60)
    
    if level == DEBUG:
        terminal_prefix = f"\033[34m[{duration} DEBUG]\033[0m"
        file_prefix = f"[{duration} DEBUG]"
    elif level == INFO:
        terminal_prefix = f"\033[32m[{duration} INFO]\033[0m"
        file_prefix = f"[{duration} INFO]"
    elif level == WARNING:
        terminal_prefix = f"\033[1;33m[{duration} WARNING]\033[0m"
        file_prefix = f"[{duration} WARNING]"
    elif level == ERROR:
        terminal_prefix = f"\033[1;31m[{duration} ERROR]\033[0m"
        file_prefix = f"[{duration} ERROR]"
    else:
        raise AssertionError 
        
    lines = str(message).split('\n')
    
    for i in range(1, len(lines)):
        lines[i] = ' ' * (len(file_prefix) + 1) + lines[i]
        
    message = ' ' + '\n'.join(lines)
        
    if level <= verbose:
        if terminal_ptr: 
            print(terminal_prefix + message, file=terminal_ptr, flush=True)
            
        if file_ptr: 
            print(file_prefix + message, file=file_ptr, flush=True)


def log_debug(message: Any):
    log(message, DEBUG)


def log_info(message: Any):
    log(message, INFO)
    
    
def log_warning(message: Any):
    log(message, WARNING)
    
    
def log_error(message: Any):
    log(message, ERROR)
