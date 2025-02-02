from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable("http://127.0.0.1:8088/chain-demo")
    print(client.invoke({
        "language": "English",
        "text": "快速的棕色狐狸跳过了懒惰的狗。"
    }))
