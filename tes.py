from concurrent.futures import ThreadPoolExecutor

def my_function():
    # 执行一些操作...
    print('hi')
    return 'result'

# 创建线程池
executor = ThreadPoolExecutor()

# 提交任务并获取Future对象
future = executor.submit(my_function)

# 等待任务执行完成并获取结果
future.wait()
result = future.result()

# 打印结果
print(result)  # 'result'
