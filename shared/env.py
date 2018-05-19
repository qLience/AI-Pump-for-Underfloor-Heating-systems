# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# From Aalborg University

# Importing Libraries
import socket
import sys
import struct
import array

# Environments
TL12, TL3, TL4, SETL2, SETL3, SETL4, ETL2, ETL3, ETL4 = ("tl12", "tl3", "tl4", "setl2", "setl3", "setl4", "etl2", "etl3", "etl4")


class environment:
	def __init__(self, env_decider):
		self.env_decider = env_decider
        # Connection for sender socket
		self.sendConn = 0
		self.sendHost = 'localhost'  # Symbolic name meaning all available interfaces
		self.sendPort = 50000        # Arbitrary non-privileged port
		# Connection for receiver socket
		self.recvConn = 0
		self.recvHost = 'localhost'  # Symbolic name meaning all available interfaces
		self.recvPort = 50001        # Arbitrary non-privileged port
		self.last_data = 0
	
	# Creating server Socket
	def createServerSockets(self):
		
		# Calling socket creater methods
		self.createSendServerSocket()
		self.createRecvServerSocket()

	def createSendServerSocket(self):
		"""Creating send server for the client (environment to connect to)
		This server will send actions from Python
		"""
		# Creating server Socket location
		serverSocketS = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			serverSocketS.bind((self.sendHost, self.sendPort))
		except:
			print('Bind failed.')
			sys.exit()
		print ('Socket bind complete')
		# Enable listening
		serverSocketS.listen(1)
		print ('Socket now listening')
		
		# Wait for client connection
		print ('waiting 20 seconds for response from client at sender port ',self.sendPort)
		serverSocketS.settimeout(20)
		try:
			self.sendConn, addr = serverSocketS.accept()
		except socket.timeout:
			print('No connection, program terminated')
			sys.exit()
		print ('Connected by', addr,'on sender port',self.sendPort)
		
		
		# Create socket for receiving
		
		# Creating server Socket
	def createRecvServerSocket(self):
		"""Creating receiver server for the client (environment to connect to)
		   This server will receive environment values(temperatures) from Python
		"""
		# Creating server Socket location
		serverSocketR = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			serverSocketR.bind((self.recvHost, self.recvPort))
		except:
			print('Bind failed.')
			sys.exit()
		print ('Socket bind complete')
		# Enable listening
		serverSocketR.listen(1)
		print ('Socket now listening')
		
		# Wait for client connection
		print ('waiting for response from client at receiver port ',self.recvPort)
		self.recvConn, addr = serverSocketR.accept()
		print ('Connected by', addr,'on receiver port',self.recvPort)
	
	def receiveState(self):
		"""Returns environment values"""
        # Receive state formed as binary array
		data = self.recvConn.recv(2048);
		# decode state
		if self.env_decider == TL12 or self.env_decider == TL3 or self.env_decider == TL4 or self.env_decider == SETL2 or self.env_decider == self.SETL3 or self.env_decider == SETL4:
			return self.decodeSimulinkState(data)
		else:
			return self.decodeMatlabState(data)
        
	def decodeMatlabState(self, data):
		"""Returns a sorted array of environment values which was received from matlab"""
        # Unpack from hex (binary array) to double
		try:
			data = str(data)
			data = data.split(",")
			del data[0]
			del data[6]
			data = [float(i) for i in data]
		except: 
			data = self.last_data
        
		return data

	def decodeSimulinkState(self, data):
		"""Returns a sorted array of environment values which was received from simulink"""
        # Unpack from hex (binary array) to double
		try:
			data = array.array('d',data)
		except: 
			data = self.last_data

		return data
		
	def sendAction(self, msg):
		"""sends a package encoded (utf-8) to environment"""
		msg = struct.pack("I",msg)
		self.sendConn.sendall(msg)#.encode('utf-8'))	