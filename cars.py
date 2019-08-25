from restful_service.restful_master_service import *

if __name__ == "__main__":
    # 23450 is http TCP port, need to open this port to external ip
    app.run(host='0.0.0.0', port=5050)
    # app.run()