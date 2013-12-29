from vbench.benchmark import Benchmark
from datetime import datetime

common_setup = """from pandas_vb_common import *
from pandas.parallel.engines import create_parallel_engine
python_engine=create_parallel_engine(name='python',force=True)
joblib_engine_1=create_parallel_engine(name='joblib',force=True,max_cpu=1)
joblib_engine_2=create_parallel_engine(name='joblib',force=True,max_cpu=2)
joblib_engine_4=create_parallel_engine(name='joblib',force=True,max_cpu=4)
joblib_engine_8=create_parallel_engine(name='joblib',force=True,max_cpu=8)
df1  = DataFrame(np.random.randn(20000, 1000))
def f1(x):
    result = [ np.sqrt(x) for i in range(10) ]
    return result[-1]
"""

SECTION = 'Parallel ops'

#----------------------------------------------------------------------
# parallel ops

setup = common_setup + """
"""
parallel_base = \
    Benchmark("df1.apply(f1,engine=python_engine)", setup, name='parallel_base',
              start_date=datetime(2013, 12, 30))

setup = common_setup + """
"""
parallel_base_1 = \
    Benchmark("df1.apply(f1,engine=joblib_engine_1)", setup, name='parallel_base_1',
              start_date=datetime(2013, 12, 30))

setup = common_setup + """
"""
parallel_base_2 = \
    Benchmark("df1.apply(f,engine=joblib_engine_2)", setup, name='parallel_base_2',
              start_date=datetime(2013, 12, 30))

setup = common_setup + """
"""
parallel_base_4 = \
    Benchmark("df1.apply(f1,engine=joblib_engine_4)", setup, name='parallel_base_4',
              start_date=datetime(2013, 12, 30))

setup = common_setup + """
"""
parallel_base_8 = \
    Benchmark("df1.apply(f1,engine=joblib_engine_8)", setup, name='parallel_base_8',
              start_date=datetime(2013, 12, 30))
