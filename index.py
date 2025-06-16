from main import Forecast

def handler(event, context):
    # 这里是定时任务的业务逻辑
    f = Forecast()
    f.run()
    return "Daily job executed successfully!"