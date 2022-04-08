"""
Adapted from  
https://stackoverflow.com/questions/32163436/python-decorator-for-printing-every-line-executed-by-a-function
"""

import sys
import tensorflow as tf

class debug_context():
    """ Debug context to trace any function calls inside the context """
    def __init__(self, name, tensor_name, run=False):
        """Does nothing when run=False."""
        self.name = name
        self.tensor_name = tensor_name
        self.run = run

    def __enter__(self):
        if self.run:
            print('Entering Debug Decorated func')
        # Set the trace function to the trace_calls function
        # So all events are now traced
        tc = self.trace_calls
        try:
            sys.settrace(tc)
        except TypeError:
            pass
        self.is_first_time = False

    def __exit__(self, *args, **kwargs):
        # Stop tracing all events
        sys.settrace = None

    def trace_calls(self, frame, event, arg): 
        if not self.run:
            return

        # We want to only trace our call to the decorated function        
        if event != 'call':
            return
        elif frame.f_code.co_name != self.name:
            return
        # return the trace function to use when you go into that 
        # function call
        return self.print_shape

    def print_shape(self, frame, event, arg):
        if event not in ['line', 'return']:
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        filename = co.co_filename
        local_vars = frame.f_locals
        for k,v in local_vars.items():
            if k == self.tensor_name and tf.is_tensor(v):
                print(v.shape)
      
def debug_tensor_shape(name, run=False):
    """ Debug decorator to print the shape of all tensors in the function. """
    def inner(func):
        def decorated_func(*args, **kwargs):
            with debug_context(func.__name__, name, run):
                return_value = func(*args, **kwargs)
            return return_value
        return decorated_func
    return inner
