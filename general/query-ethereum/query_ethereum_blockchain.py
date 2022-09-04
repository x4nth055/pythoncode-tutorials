from web3 import Web3

# infura API key
API_KEY = "put your API key here"
# change endpoint to mainnet or ropsten or any other of your account
url = f"https://<endpoint>.infura.io/v3/{API_KEY}"

w3 = Web3(Web3.HTTPProvider(url))
# see whether the connection is established
res = w3.isConnected()
print(res)


# get latest block
latest = w3.eth.get_block('latest')
print(latest)
# print the block number
print(latest['number'])


# query individual transactions
transaction1 = w3.eth.get_transaction('0x0e3d45ec3e1d145842ce5bc56ad168e4a98508e0429da96c1ff89f11076da36d')
print(transaction1)


# use block number to query transactions
transaction2 = w3.eth.get_transaction_by_block(15410924, 0)
print(transaction2)


# get number of transactions in a block
transactionCount = w3.eth.get_transaction_count('0x486976656f6e2065752d68656176792d657163')
print(transactionCount)


# check if a block address is valid
isValid = w3.isAddress('0xed44e77fb3408cd5ad415d7467af6f6783218fb74c3824de1258f6d266bcc7b7')
print(isValid)


# check if an address is a valid EIP-55 checksum address
isChecksumAddressValid = Web3.isChecksumAddress('0x486976656f6e2065752d68656176792d657163')
print(isChecksumAddressValid)


# get balance of a block address
balance = w3.eth.get_balance('0xd3CdA913deB6f67967B99D67aCDFa1712C293601')
print(balance)


# get proof of a block
proof = w3.eth.get_proof('0x486976656f6e2065752d68656176792d657163', [0], 3391)
print(proof)


# get uncle of a block
w3.eth.get_uncle_by_block(15410924, 0)


nonce = w3.eth.getTransactionCount('0x610Ae88399fc1687FA7530Aac28eC2539c7d6d63', 'latest')
# create a transaction
transaction = {
     'to': '0x31B98D14007bDEe637298086988A0bBd31184523', 
     'from': '0x31B98D14007bDEe63EREEDFT34544646MOI22',
     'value': 500,
     'gas': 10000,
     'maxFeePerGas': 1000000208,
     'nonce': nonce,
}
# send the transaction
w3.eth.send_transaction(transaction)


# sign a transaction
signed = w3.eth.sign_transaction(
    dict(
        nonce=nonce,
        maxFeePerGas=34300000,
        maxPriorityFeePerGas=25000000,
        gas=100000,
        to='0xerecfBYWlB99D67aCDFa17EREFEerrtr73601',
        value=1,
        data=b'',
    )
)

address = '0x706f6f6c696e2e636f6d21688947c8f76c4e92'
abi = '[{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"address","name":"minter_",........' # abi of the contract
# create a contract object
contract = w3.eth.contract(address=address, abi=abi)


# total supply of the token
totalSupply = contract.functions.totalSupply().call()
print(totalSupply)


# read the data and update the state
contract.functions.storedvalue().call()
tx_hash = contract.functions.updateValue(100).transact()


# retrieve token metadata
print(contract.functions.name().call())
print(contract.functions.decimals().call())
print(contract.functions.symbol().call())
# output:
# SHIBACHU
# 9
# SHIBACHU


# find the account balance
address = '0x5eaaf114aad1313e7440d2ff805ced993e566df'
balance = contract.functions.balanceOf(address).call()

