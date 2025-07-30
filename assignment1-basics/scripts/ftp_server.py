import os
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer

DATA_DIR = os.path.abspath("data")

authorizer = DummyAuthorizer()
authorizer.add_anonymous(DATA_DIR, perm="elradfmwMT")

handler = FTPHandler
handler.authorizer = authorizer
handler.dtp_handler = ThrottledDTPHandler

# Listen on all IPv6 interfaces, port 2121
address = ("::", 2121)

server = FTPServer(address, handler)
server.serve_forever()
