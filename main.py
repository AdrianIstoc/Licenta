from MainWindow.main_window import MainWindow
from SentinelHub.sh_connection import SentinelHubConnection

def main():
    connection = SentinelHubConnection()
    config = connection.get_config()

    app = MainWindow(config=config)
    app.mainloop()

if __name__ == "__main__":
    main()