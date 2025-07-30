import tempfile


class TempStringFile:
    def __init__(self, content, mode="w+", encoding="utf-8"):
        self.content = content
        self.mode = mode
        self.encoding = encoding
        self.file = None

    def __enter__(self):
        self.file = tempfile.NamedTemporaryFile(self.mode, encoding=self.encoding, delete=True)
        self.file.write(self.content)
        self.file.flush()
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            self.file = None
